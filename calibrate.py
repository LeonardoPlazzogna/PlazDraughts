"""
Calibration of the SEARCH hyperparameters by direct strength comparison.

The very same network plays itself with two different search settings,
alternating White. Whoever wins has the better search: it is a measurement of
PLAYING STRENGTH, not of some indirect property.

WHY IT IS NEEDED, with an example that cost little only because it was measured.
The self-play metrics showed an apparently clear signature: q_spread growing
while the entropy of the target stayed high, about 2.9 effective moves out of
5.69 legal ones. From that one would conclude that the search spread its visits
too thin and that exploration should be LOWERED.

The measurement said otherwise: lowering c_puct loses strength (1.0 scores
0.450 [0.413, 0.487] against 1.5 over 400 games), while 2.0 and 3.0 cannot be
told apart from 1.5 (docs/results.md, section 2.1). The sharpness of the target
does not predict playing strength, and no quantity observed during self-play
replaces a strength comparison.

A first run of this very tool had given c_puct 0.5 at -139 Elo and 2.0 at +40,
monotonically across the grid. Those 400 games contained about 16 distinct
games, and the trend was an artifact (section 5.1): it is what the null check
below exists to catch.

That is why the tool exists: not to confirm an intuition, but to be able to
refute it.

USAGE. With the C++ engine (recommended: parallel, at the real parameters):
    DAMA_CPP_ENGINE=./engine_c/build/dama_engine DAMA_DEVICE=cuda python calibrate.py

Without the engine it falls back on the in-process Python path: correct but
sequential, usable only with few games and few simulations.

    DAMA_CAL_GAMES=400   games per comparison
    DAMA_CAL_SIMS=400    simulations (the same as the real run)
    DAMA_CAL_GRID=...    values to try
    DAMA_CAL_REF=1.5     reference (the run's current default)
    DAMA_CAL_SEED=4242   seed; changing it and running again is the most direct
                         way to see how noisy the measurement really is
    DAMA_CAL_TEMP_PLIES=10  sampled opening plies (see below)
    DAMA_CAL_NULL=1      run the null check before the grid

THE NULL CHECK, and why it is there. Without Dirichlet noise and with a
deterministic network, two games that start with the same moves are THE SAME
GAME. Diversity comes only from the first plies, sampled from the visits: if
they are few, four hundred games contain far fewer than FOUR HUNDRED independent
observations, and a confidence interval computed on the number of games played
comes out too narrow -- differences that do not exist get declared.

The check makes the reference play AGAINST ITSELF. The true result is 0.500 by
construction: how far the measurement is from it, and whether 0.500 falls inside
the interval the measurement itself declares, tells whether the grid's intervals
can be trusted. It is the only way to find out that a measuring tool lies before
believing it.

A TRAINED network in weights/champion.pt is needed. With a random network the
priors are noise, c_puct changes almost nothing and the result means nothing:
the program says so and continues only as a smoke test.
"""
from __future__ import annotations
import math
import os
import shutil
import time

import numpy as np
import torch

from dama import IN_PLANES, Position
from model import PolicyValueNet
from evaluators import NetEvaluator
from mcts import MCTS
import cpp_engine
from sprt import score_ci                     # one implementation, shared

# The same defaults as the conductor: the calibration must run on the network and
# the parameters of the real run, otherwise it measures something else. (A
# calibration built with different defaults -- a 64x5 network against the
# conductor's 96x6 -- cannot even load the champion.)
CHANNELS   = int(os.environ.get("DAMA_CHANNELS", "96"))
N_BLOCKS   = int(os.environ.get("DAMA_BLOCKS", "6"))
USE_SE     = os.environ.get("DAMA_USE_SE", "1") != "0"
GAMES      = int(os.environ.get("DAMA_CAL_GAMES", "400"))
SIMS       = int(os.environ.get("DAMA_CAL_SIMS", "400"))
BATCH      = int(os.environ.get("DAMA_MCTS_BATCH", "8"))
# Which search parameter to calibrate. Both are measured the same way -- the
# same network on both sides, the parameter different -- but they answer
# different questions: c_puct asks which balance between prior and value plays
# better for the same budget, n_sims asks whether more budget is worth its cost.
PARAM      = os.environ.get("DAMA_CAL_PARAM", "c_puct")
if PARAM not in ("c_puct", "n_sims"):
    raise SystemExit("DAMA_CAL_PARAM must be c_puct or n_sims")

_DEF_REF  = {"c_puct": "1.5", "n_sims": str(SIMS)}[PARAM]
_DEF_GRID = {"c_puct": "0.5,0.75,1.0,1.25,2.0",
             "n_sims": "100,200,800,1600"}[PARAM]
REF        = float(os.environ.get("DAMA_CAL_REF", _DEF_REF))
GRID       = [float(x) for x in os.environ.get(
    "DAMA_CAL_GRID", _DEF_GRID).split(",") if x.strip()]
WEIGHTS    = os.environ.get("DAMA_WEIGHTS_DIR", "weights")
ENGINE     = os.environ.get("DAMA_CPP_ENGINE", "")
DEVICE     = os.environ.get("DAMA_DEVICE", "cpu")
THREADS    = int(os.environ.get("DAMA_CPP_THREADS", str(os.cpu_count() or 8)))
MAX_BATCH  = int(os.environ.get("DAMA_SERVER_BATCH", "256"))
# c_puct used by both sides when n_sims is calibrated.
REF_CPUCT  = float(os.environ.get("DAMA_C_PUCT", "1.5"))
SEED       = int(os.environ.get("DAMA_CAL_SEED", "4242"))
# Opening plies sampled instead of played greedily -- the same number the arena
# uses, and for a sharper reason. There two DIFFERENT networks meet and diverge
# on their own; here the network is THE SAME on both sides and the only
# difference is a search parameter, so the games stay together far longer.
# Without varied openings the same few distinct games get replayed, and the
# interval would be computed on repetitions.
TEMP_PLIES = int(os.environ.get("DAMA_CAL_TEMP_PLIES", "10"))
NULL_CHECK = os.environ.get("DAMA_CAL_NULL", "1") != "0"


def elo(score: float) -> float:
    s = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


# --- C++ path: parallel, at the real parameters ------------------------------
def match_cpp(net, v_a: float, v_b: float, n_games: int, seed: int, workdir: str):
    """Side A with value v_a, side B with value v_b, everything else identical."""
    if PARAM == "c_puct":
        kw = dict(n_sims=SIMS, c_puct=v_a, c_puct_b=v_b)
    else:
        kw = dict(n_sims=int(v_a), c_puct=REF_CPUCT, n_sims_b=int(v_b))
    _, (w, d, l), out = cpp_engine.arena(
        ENGINE, net, net, n_games=n_games, n_threads=THREADS,
        device=DEVICE, mcts_batch=BATCH, max_batch=MAX_BATCH, seed=seed,
        in_planes=IN_PLANES, temp_plies=TEMP_PLIES, workdir=workdir, **kw)
    return w, d, l, cpp_engine.parse_arena_distinct(out)


# --- Python path: sequential, a fallback -------------------------------------
def match_py(net, v_a: float, v_b: float, n_games: int, seed: int, _wd=None):
    ev = NetEvaluator(net, DEVICE)
    w = d = l = 0
    for i in range(n_games):
        a_white = (i % 2 == 0)
        pos, plies = Position(), 0
        rng = np.random.default_rng(seed + i)
        while not pos.is_terminal() and plies < 300:
            mine = (pos.turn == 1) == a_white
            val = v_a if mine else v_b
            c = val if PARAM == "c_puct" else REF_CPUCT
            ns = SIMS if PARAM == "c_puct" else int(val)
            root = MCTS(ev, n_sims=ns, c_puct=c, batch_size=BATCH,
                        rng=rng).run(pos, add_noise=False)
            moves = list(root.children.items())
            if plies < TEMP_PLIES:
                # Opening sampled from the visits, as in the C++ engine. Without
                # it every game of this loop would be identical to the others
                # with the same colors -- the network is deterministic and there
                # is no noise -- and the n_games games would carry the
                # information of just two.
                p = np.array([kv[1].N for kv in moves], dtype=np.float64)
                move = moves[int(rng.choice(len(moves), p=p / p.sum()))][0]
            else:
                move = max(moves, key=lambda kv: kv[1].N)[0]
            pos = pos.play(move)
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
    ckpt = os.path.join(WEIGHTS, "champion.pt")
    net = PolicyValueNet(in_planes=IN_PLANES, channels=CHANNELS,
                                 n_blocks=N_BLOCKS, use_se=USE_SE)
    trained = os.path.exists(ckpt)
    if trained:
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        print(f"[calibrate] trained network loaded from {ckpt}")
    else:
        print(f"[calibrate] WARNING: {ckpt} does not exist -> RANDOM network.")
        print("            With random priors the parameter changes almost nothing: the")
        print("            result is NOT usable, it only serves to try the tool.")
    net.eval()

    use_cpp = bool(ENGINE) and os.path.exists(ENGINE)
    if ENGINE and not use_cpp:
        print(f"[calibrate] C++ engine not found at {ENGINE}: Python path")
    run = match_cpp if use_cpp else match_py
    workdir = os.path.join(WEIGHTS, f"cal_tmp_{os.getpid()}")

    via = (f"C++ engine ({THREADS} threads, {DEVICE})" if use_cpp
           else "sequential Python")
    print(f"[calibrate] network {CHANNELS}x{N_BLOCKS}, {SIMS} simulations, "
          f"{GAMES} games per comparison, reference {PARAM} = {REF:g}")
    print(f"[calibrate] path: {via}")
    print(f"[calibrate] seed {SEED}, {TEMP_PLIES} sampled opening plies")
    print()

    def measure(v_a, v_b):
        t0 = time.time()
        w, d, l, dist = run(net, v_a, v_b, GAMES, SEED, workdir)
        s, half = score_ci(w, d, l, dist)
        return s, half, w, d, l, time.time() - t0, dist

    results = []
    suspect = False
    try:
        # --- null check: the reference against itself ------------------------
        # The true result is 0.500 by construction. If it does not fall inside
        # the declared interval, the interval is wrong, and with it every verdict
        # of the grid. It must be measured BEFORE looking at the results,
        # otherwise one ends up believing the row that confirms what was hoped.
        if NULL_CHECK:
            s0, h0, w0, d0, l0, dt0, dist0 = measure(REF, REF)
            print("  NULL CHECK -- the reference against itself")
            print(f"    score {s0:.3f} [{s0 - h0:.3f}, {s0 + h0:.3f}]   "
                  f"({w0} won, {d0} drawn, {l0} lost, {dt0:.0f}s)")
            if dist0 is not None:
                print(f"    different games: {dist0} of {GAMES}"
                      + ("" if dist0 >= GAMES * 0.9 else
                         "  <-- the others are repetitions, and count as one"))
            suspect = not (s0 - h0 <= 0.5 <= s0 + h0)
            if suspect:
                print("    0.500 does NOT fall inside the interval: the intervals are")
                print("    TOO NARROW and the verdicts below are not valid.")
            else:
                print("    0.500 falls inside the interval: the measurement is not biased")
                print(f"    (observed deviation from the truth: {abs(s0 - 0.5):.3f}).")
            print()

        print(f"  {PARAM:<8} score (95% CI)                  Elo vs reference       verdict")
        print("  " + "-" * 76)
        for c in GRID:
            if abs(c - REF) < 1e-9:
                continue
            s, half, w, d, l, dt, dist = measure(c, REF)
            e = elo(s)
            e_lo, e_hi = elo(max(1e-6, s - half)), elo(min(1.0 - 1e-6, s + half))
            # The verdict looks at the INTERVAL, not at the point estimate. A
            # score of 0.53 over 400 games is indistinguishable from 0.50, and
            # treating it as an improvement is exactly the mistake that made the
            # Elo ladder of the previous run unreadable.
            if s - half > 0.5:
                verdict = "BETTER than the reference"
            elif s + half < 0.5:
                verdict = "worse"
            else:
                verdict = "indistinguishable"
            print(f"  {c:<7.2f}  {s:.3f} [{s - half:.3f}, {s + half:.3f}]   "
                  f"{e:+7.1f} [{e_lo:+7.1f}, {e_hi:+7.1f}]  {verdict}")
            different = "" if dist is None else f", {dist} different"
            print(f"           {w} won, {d} drawn, {l} lost{different}   "
                  f"({dt:.0f}s)")
            results.append((c, s, half, verdict, dist or GAMES))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if suspect:
        print("  DO NOT ADOPT ANY VALUE FROM THIS TABLE.")
        print("  The null check failed: the measurement declares intervals")
        print("  narrower than the error it really makes. First fix the tool")
        print("  (more DAMA_CAL_TEMP_PLIES, more DAMA_CAL_GAMES),")
        print("  then look at the numbers.")
        return

    better = [r for r in results if r[3].startswith("BETTER")]
    if better:
        best = max(better, key=lambda r: r[1])
        print(f"  Best: {PARAM}={best[0]:g} (score {best[1]:.3f}).")
        # A maximum on the EDGE of the grid is not a maximum: it is the point
        # beyond which nobody looked. Adopting it means mistaking for an optimum
        # the limit of an arbitrary choice made before measuring.
        if best[0] >= max(GRID) - 1e-9 or best[0] <= min(GRID) + 1e-9:
            print(f"  WARNING: {best[0]:g} is an EDGE of the grid, so the optimum")
            print("  is not bracketed: the result says \"at least this much\",")
            print("  not \"exactly this much\". Extend the grid on that side")
            print("  before adopting it.")
        env = "DAMA_C_PUCT" if PARAM == "c_puct" else "DAMA_SIMS"
        print(f"  To adopt it in the run:  {env}={best[0]:g} ./run.sh")
        print(f"  But first repeat with another seed (DAMA_CAL_SEED) and check")
        print("  that the winner stays the same: a ranking that changes with the")
        print("  seed is noise, not a difference in strength.")
        if PARAM == "n_sims":
            print("  NOTE: more simulations cost time in proportion. The best value")
            print("  here is the STRONGEST, not the most convenient: to choose,")
            print("  compare the Elo gain with the cost per cycle.")
    else:
        n_ind = min([r[4] for r in results], default=GAMES)
        threshold = elo(0.5 + 1.96 * math.sqrt(0.125 / max(1, n_ind))) if n_ind else 0
        print(f"  No value beats {REF} distinguishably from noise.")
        print(f"  With {n_ind} different games, differences from ~{threshold:.0f} Elo up can be told apart:")
        print("  if smaller ones exist, more games are needed (DAMA_CAL_GAMES).")
    if not trained:
        print("\n  (random network: the result above has no value)")


if __name__ == "__main__":
    main()
