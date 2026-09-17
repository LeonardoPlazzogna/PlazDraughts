"""
Direct comparison between TWO different networks, with the same search.

It complements calibrate.py, and the difference between the two is the one
between the two classes of hyperparameters:

  * calibrate.py varies a SEARCH parameter with a single network on both sides.
    It answers questions such as "which c_puct plays better" without training
    anything;

  * compare.py pits TWO already trained networks against each other. It is for
    everything decided during training -- learning rate, weight of the value
    term, epochs per cycle, width and depth of the network -- because the effect
    of those parameters exists only after someone has trained with them. The
    protocol is always the same: two short runs starting from the same champion
    and differing in a single parameter, then this tool on the two resulting
    champions.

The search is identical on both sides: what is measured is the difference
between the networks, not between ways of searching.

USAGE
    DAMA_CPP_ENGINE=./engine_c/build/dama_engine DAMA_DEVICE=cuda \\
    python tools/compare.py weights_A.pt weights_B.pt

    DAMA_CMP_GAMES=400   games (default 400)
    DAMA_CMP_SIMS=400    simulations per move
    DAMA_CMP_TEMP_PLIES=10  opening plies sampled instead of greedy. Without
                         noise and with deterministic networks, two games that
                         start the same ARE the same game: the distinct games
                         are at most as many as the distinct openings, and an
                         interval computed on the number of games played would
                         be narrower than it really is.
    DAMA_CHANNELS/DAMA_BLOCKS/DAMA_USE_SE  architecture of the two networks
                                           (they must share it)

The verdict looks at the confidence INTERVAL, not at the score: 0.53 over 400
games is indistinguishable from 0.50, and treating it as an improvement is the
most common way of chasing noise for days.
"""
from __future__ import annotations
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import IN_PLANES, Position
from model import PolicyValueNet
from evaluators import NetEvaluator
from mcts import MCTS
from sprt import score_ci                     # one implementation, shared
import cpp_engine

CHANNELS  = int(os.environ.get("DAMA_CHANNELS", "96"))
N_BLOCKS  = int(os.environ.get("DAMA_BLOCKS", "6"))
USE_SE    = os.environ.get("DAMA_USE_SE", "1") != "0"
GAMES     = int(os.environ.get("DAMA_CMP_GAMES", "400"))
SIMS      = int(os.environ.get("DAMA_CMP_SIMS", "400"))
BATCH     = int(os.environ.get("DAMA_MCTS_BATCH", "8"))
C_PUCT    = float(os.environ.get("DAMA_C_PUCT", "1.5"))
ENGINE    = os.environ.get("DAMA_CPP_ENGINE", "")
DEVICE    = os.environ.get("DAMA_DEVICE", "cpu")
THREADS   = int(os.environ.get("DAMA_CPP_THREADS", str(os.cpu_count() or 8)))
MAX_BATCH = int(os.environ.get("DAMA_SERVER_BATCH", "256"))
SEED      = int(os.environ.get("DAMA_CMP_SEED", "20260801"))
TEMP_PLIES = int(os.environ.get("DAMA_CMP_TEMP_PLIES", "10"))


def elo(score: float) -> float:
    s = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def load(path: str):
    net = PolicyValueNet(in_planes=IN_PLANES, channels=CHANNELS,
                                 n_blocks=N_BLOCKS, use_se=USE_SE)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def match_cpp(net_a, net_b, workdir: str):
    _, (w, d, l), out = cpp_engine.arena(
        ENGINE, net_a, net_b, n_games=GAMES, n_sims=SIMS, n_threads=THREADS,
        device=DEVICE, mcts_batch=BATCH, max_batch=MAX_BATCH, seed=SEED,
        in_planes=IN_PLANES, c_puct=C_PUCT, temp_plies=TEMP_PLIES,
        workdir=workdir)
    return w, d, l, cpp_engine.parse_arena_distinct(out)


def match_py(net_a, net_b, _wd=None):
    ea, eb = NetEvaluator(net_a, DEVICE), NetEvaluator(net_b, DEVICE)
    w = d = l = 0
    for i in range(GAMES):
        a_white = (i % 2 == 0)
        pos, plies = Position(), 0
        rng = np.random.default_rng(SEED + i)
        while not pos.is_terminal() and plies < 300:
            ev = ea if ((pos.turn == 1) == a_white) else eb
            root = MCTS(ev, n_sims=SIMS, c_puct=C_PUCT, batch_size=BATCH,
                        rng=rng).run(pos, add_noise=False)
            moves = list(root.children.items())
            if plies < TEMP_PLIES:
                p = np.array([kv[1].N for kv in moves], dtype=np.float64)
                pos = pos.play(moves[int(rng.choice(len(moves), p=p / p.sum()))][0])
            else:
                pos = pos.play(max(moves, key=lambda kv: kv[1].N)[0])
            plies += 1
        r = pos.result() or 0
        ar = r if a_white else -r
        w += (ar > 0)
        d += (ar == 0)
        l += (ar < 0)
    return w, d, l, None


def main():
    # A --help that answers: these tools take no arguments, so without this
    # line they would start computing (see env_help.py).
    from env_help import help_if_requested
    help_if_requested(__doc__, __file__)
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    pa, pb = sys.argv[1], sys.argv[2]
    for p in (pa, pb):
        if not os.path.exists(p):
            raise SystemExit(f"file not found: {p}")

    net_a, net_b = load(pa), load(pb)
    use_cpp = bool(ENGINE) and os.path.exists(ENGINE)
    if ENGINE and not use_cpp:
        print(f"[compare] C++ engine not found at {ENGINE}: Python path")
    workdir = os.path.join(os.environ.get("DAMA_WEIGHTS_DIR", "weights"),
                           f"cmp_tmp_{os.getpid()}")

    via = (f"C++ engine ({THREADS} threads, {DEVICE})" if use_cpp
           else "sequential Python")
    print(f"[compare] A = {pa}")
    print(f"[compare] B = {pb}")
    print(f"[compare] network {CHANNELS}x{N_BLOCKS}, {SIMS} simulations, "
          f"c_puct {C_PUCT}, {GAMES} games, alternating colors, "
          f"{TEMP_PLIES} sampled opening plies (seed {SEED})")
    print(f"[compare] path: {via}\n")

    t0 = time.time()
    try:
        w, d, l, dist = (match_cpp(net_a, net_b, workdir) if use_cpp
                         else match_py(net_a, net_b))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    s, half = score_ci(w, d, l, dist)
    e = elo(s)
    e_lo, e_hi = elo(max(1e-6, s - half)), elo(min(1.0 - 1e-6, s + half))
    if s - half > 0.5:
        verdict = "A is STRONGER than B"
    elif s + half < 0.5:
        verdict = "A is weaker than B"
    else:
        verdict = "indistinguishable"

    print(f"  {w} won, {d} drawn, {l} lost   ({time.time() - t0:.0f}s)")
    if dist is not None:
        print(f"  different games: {dist} of {GAMES}"
              + ("" if dist >= GAMES * 0.9 else
                 "  <-- the interval below is computed on these"))
    print(f"  score of A     : {s:.3f}  [{s - half:.3f}, {s + half:.3f}]")
    print(f"  Elo of A vs B  : {e:+.1f}  [{e_lo:+.1f}, {e_hi:+.1f}]")
    print(f"\n  {verdict}")
    if verdict == "indistinguishable":
        n_ind = dist or GAMES
        threshold = elo(0.5 + 1.96 * math.sqrt(0.125 / max(1, n_ind)))
        print(f"  With {n_ind} different games, differences from "
              f"~{threshold:.0f} Elo up can be told apart.")
        print("  An indistinguishable result does NOT mean that the two networks")
        print("  are equal: it means that this measurement is not enough to say.")


if __name__ == "__main__":
    main()
