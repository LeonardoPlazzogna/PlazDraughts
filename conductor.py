"""
Conductor of the AlphaZero pipeline for Italian draughts.

It follows the loop of the chess project:

  for each cycle:
    1) SELF-PLAY: N games with the champion network -> samples into the buffer
                  (C++ engine if DAMA_CPP_ENGINE is set, otherwise Python)
    2) TRAIN    : trains a copy of the champion on the replay buffer
    3) ARENA    : the trained candidate plays the champion; if the SPRT (or the
                  fixed threshold) accepts it, it is PROMOTED to champion
    4) saves the weights and logs the cycle summary
    5) every STRENGTH_EVERY cycles: STRENGTH TEST against fixed opponents
       (fixed-depth alpha-beta by default); every GEN_EVERY cycles: the
       generation ladder (ladder.py), the champion against its predecessors
    6) writes the resume state (last completed cycle + replay buffer) to disk,
       atomically.

RESUME: at startup, if weights/champion.pt and weights/train_state.pkl exist,
the champion's weights, the replay buffer and the number of the last completed
cycle are reloaded, and the run resumes from the next cycle instead of starting
again from 1. If train_state.pkl is missing or corrupt the run starts from cycle
1 with an empty buffer (the champion's weights, if present, are still the
strongest found so far).

Parameters can be overridden through environment variables (like CONDUCTOR_* in
the chess project), for quick tests without touching the code.
"""
from __future__ import annotations
import glob
import math
import os
import pickle
import shutil
import time
from collections import deque

import numpy as np
import torch

from dama import IN_PLANES, Position
from model import PolicyValueNet
from evaluators import NetEvaluator
from selfplay import play_game
from train import train_on_samples
from players import RandomPlayer, GreedyPlayer, AlphaBetaPlayer
from metrics import MetricsCsv, CycleRow, aggregate
from parallel import WorkerPool
from inference_server import InferenceServer
import cpp_engine
from ladder import Ladder
from runlock import acquire, RunLockError
import sprt

# --- parameters (real defaults; quick tests via env) -------------------------
CYCLES         = int(os.environ.get("DAMA_CYCLES", "100"))
# Self-play games per cycle. It has to be large enough for one cycle to renew a
# good share of the replay buffer: if candidate and champion are trained on
# almost the same samples they are almost the same network, the arena cannot
# tell them apart, and what gets promoted is noise.
GAMES_PER_CYCLE = int(os.environ.get("DAMA_GAMES", "400"))
N_SIMS         = int(os.environ.get("DAMA_SIMS", "400"))
# Final SELF-PLAY simulations, interpolated linearly from N_SIMS over the run.
# Default = N_SIMS, i.e. no schedule.
#
# Why schedule them instead of raising them at once. Once the network follows
# the visit distribution closely, what limits the target is that the
# distribution itself is DIFFUSE, and only more simulations sharpen it -- at a
# cost that grows linearly. In the first cycles the value head is still crude
# and extra simulations pay little, so the budget is spent where it helps,
# towards the end. tools/diagnose_targets.py says which of the two regimes a
# run is in.
#
# The EVALUATION simulations (arena, ladder, strength against baselines) stay
# fixed at N_SIMS on purpose: if they changed over time, the generational Elo
# curve and the comparisons with the anchors would no longer be homogeneous
# across cycles, and they would measure the search budget instead of the
# network's strength.
SIMS_FINAL     = int(os.environ.get("DAMA_SIMS_FINAL", str(N_SIMS)))
ARENA_GAMES    = int(os.environ.get("DAMA_ARENA_GAMES", "200"))
# Opening plies sampled from the visits instead of played greedily.
#
# It decides how many arena games are REALLY different, and it matters much
# more than it seems. The arena uses no Dirichlet noise and the networks are
# deterministic: two games that start with the same moves continue identically
# to the end. The distinct games possible are as many as the distinct openings,
# and playing more games adds no information -- it replays those already there.
#
# It matters because the SPRT computes the evidence on the number of games
# PLAYED. If only a fifth of them differ, the likelihood ratio is inflated by the
# same factor and the promotion gate opens on evidence it does not have.
#
# The default is generous, because with a trained network the visits are much
# more concentrated than with an untrained one and the same threshold yields
# fewer distinct games. It costs nothing: the plies are played anyway, the only
# change is whether the most visited move is chosen or the visits are sampled.
# game_diversity.py counts how many distinct games a given value produces.
ARENA_TEMP_PLIES = int(os.environ.get("DAMA_ARENA_TEMP_PLIES", "10"))
CHANNELS       = int(os.environ.get("DAMA_CHANNELS", "96"))
N_BLOCKS       = int(os.environ.get("DAMA_BLOCKS", "6"))
EPOCHS         = int(os.environ.get("DAMA_EPOCHS", "2"))
BATCH_SIZE     = int(os.environ.get("DAMA_BATCH", "1024"))
LR             = float(os.environ.get("DAMA_LR", "1e-3"))
# Size of the replay buffer, in SAMPLES (not games).
#
# What matters is the WINDOW it covers: the size divided by the samples a cycle
# produces. Too short a window and the network chases a non-stationary target
# with a very short memory, visible in a bouncing value loss; too long and it
# keeps training on the games of older, weaker champions. 600k keeps a window of
# about ten cycles at the default number of games.
#
# It is stated in samples and not in games because the two are not
# interchangeable: how many samples a game yields depends on its length, which
# grows as the network learns. A buffer sized in games silently shrinks.
#
# It costs no GPU, only memory and a larger resume checkpoint.
#
# THE RISK, stated: a longer window holds games of older champions, hence worse
# targets. If the plateau does not move but strength gets worse, this is the
# culprit and the value goes down to 400k.
BUFFER_SAMPLES = int(os.environ.get("DAMA_BUFFER_SAMPLES", "600000"))
# Discount of the value target with the distance from the end: a position d
# plies before the end gets z*gamma^d. 1.0 = off.
#
# Without a discount, in a won position every move is worth +1 and the search
# has no way to prefer one: with four kings against one, all fifteen legal moves
# are valued +1.000 and the visits spread almost evenly, so the engine shuffles
# its kings instead of converting. With the discount, winning sooner is worth
# more than winning later, and the search has a direction again.
#
# What it improves is BEHAVIOR; that it translates into strength is not shown.
# tools/ab_value_discount.py runs the comparison between the two labellings.
#
# WATCH FOR IT in the run data: the discount shrinks the positive labels and
# leaves the zeros of the draws intact, so it weakens the overall signal. If
# strength against the fixed anchors gets worse, this is the suspect and the
# value goes back to 1.0.
VALUE_DISCOUNT = float(os.environ.get("DAMA_VALUE_DISCOUNT", "0.99"))
# Learning-rate decay OVER THE RUN (on top of the cosine inside each training
# call). A fixed lr made the network oscillate around the minimum without ever
# settling. Cosine from LR to LR*DAMA_LR_FINAL over DAMA_CYCLES cycles; being a
# function of the cycle number only, it survives resumes.
LR_SCHEDULE    = os.environ.get("DAMA_LR_SCHEDULE", "1") != "0"
LR_FINAL_FRAC  = float(os.environ.get("DAMA_LR_FINAL", "0.1")) if LR_SCHEDULE else 1.0
# Weight EMA: OFF by default, because it works against this training regime.
# Every cycle is a SHORT fine-tuning that moves away from the champion's weights
# in a definite direction. Averaging a directional trajectory means lagging
# behind, so the candidate comes out weaker than the champion and the promotion
# gate rejects it. EMA helps when the trajectory oscillates around a minimum,
# not when it advances. Left as an option (DAMA_EMA=1) so that it can be
# re-evaluated if the training regime changes.
USE_EMA        = os.environ.get("DAMA_EMA", "0") != "0"
PROMOTE_MIN    = float(os.environ.get("DAMA_PROMOTE_MIN", "0.55"))
# PUCT exploration. A lower c_puct concentrates the visits and sharpens the
# policy target, roughly like giving the search more simulations, at no cost.
# CAUTION: that changes the SHARPNESS of the target, not necessarily STRENGTH.
# Less exploration can also make the search blind. The default stays 1.5 until a
# strength comparison (calibrate.py) says otherwise.
C_PUCT         = float(os.environ.get("DAMA_C_PUCT", "1.5"))
# Promotion gate via SPRT (sprt.py). The fixed-threshold gate over 30 games
# promotes ~29% of EQUIVALENT networks and rejects ~29% of REAL improvements: a
# third of the decisions are noise. Measured by Monte Carlo simulation (2000
# matches, 35% draws):
#   fixed threshold 30 : 28.8% false positives, 79.5% true positives, 30 games
#   SPRT cap 60        : 19.2% false positives, 83.8% true positives, ~57 games
#   SPRT cap 200       :  6.5% false positives, 94.8% true positives, ~173 games
# Even a cap of 60 beats the fixed gate on BOTH error types, for ~+14% of cycle
# time. The default cap is 400 games (DAMA_SPRT_MAX_GAMES). DAMA_SPRT=0 goes
# back to the fixed threshold.
USE_SPRT       = os.environ.get("DAMA_SPRT", "1") != "0"
SPRT_MAX_GAMES = int(os.environ.get("DAMA_SPRT_MAX_GAMES", "400"))
SPRT_ALPHA     = float(os.environ.get("DAMA_SPRT_ALPHA", "0.05"))
SPRT_BETA      = float(os.environ.get("DAMA_SPRT_BETA", "0.05"))
# Cadence of the fixed opponents. Each evaluation is a full match per anchor,
# and those matches compete with the cycles themselves: at a cadence of five
# cycles the run spends about as much time measuring as learning. At twenty it
# spends a quarter of it, and a hundred cycles still carry five measurement
# points -- few but each one solid, instead of many inside the noise.
#
# This is not the strength curve to watch anyway: that is the generation ladder
# below, which goes through the C++ engine and is almost free.
STRENGTH_EVERY = int(os.environ.get("DAMA_STRENGTH_EVERY", "20"))  # 0 = never
# Games per anchor. Raised from 20 to 60: with 20 games the 95% confidence
# interval on a score of 0.50 is +-111 Elo, and on 0.75 it goes from +67 to
# +391 -- wide enough to hide any change worth seeing. At 60 games the interval
# narrows by ~1.7 times.
STRENGTH_GAMES = int(os.environ.get("DAMA_STRENGTH_GAMES", "60"))
# GENERATIONAL measurement (ladder.py): the champion against the past champions.
# Unlike the fixed opponents above, it does not saturate. 0 = off.
#
# It is the meter to watch, and it is cheap: it goes through the C++ engine (in
# parallel) instead of the sequential Python path of the fixed opponents. That
# is why its games are raised much more generously.
GEN_EVERY      = int(os.environ.get("DAMA_GEN_EVERY", "10"))
GEN_GAMES      = int(os.environ.get("DAMA_GEN_GAMES", "100"))  # games per anchor
GEN_ANCHORS    = int(os.environ.get("DAMA_GEN_ANCHORS", "3"))  # how many generations back

# --- guard against champion drift ---------------------------------------------
# Games against an OLD generation before a promotion is confirmed. 0 = off.
#
# OFF BY DEFAULT, and the reason is not technical: it adds a second gate to
# every promotion, so a run with it on and a run with it off differ in more than
# the parameter under study and cannot be compared. Turning it on belongs to a
# run dedicated to it.
ANTIDRIFT      = int(os.environ.get("DAMA_ANTIDRIFT", "0"))
# How far back to look. With the ladder every 10 cycles, 3 generations are about
# 30 cycles: far enough for the reference not to have drifted along with the
# champion, not so far as to make it unbeatable and block everything.
ANTIDRIFT_BACK = int(os.environ.get("DAMA_ANTIDRIFT_BACK", "3"))
# Threshold: below it the candidate is rejected. 0.5 only asks that it does not
# REGRESS against the past, not that it dominates it.
ANTIDRIFT_MIN  = float(os.environ.get("DAMA_ANTIDRIFT_MIN", "0.5"))
# Absolute strength anchors: fixed-depth alpha-beta opponents
# (players.AlphaBetaPlayer). Empty = none.
#
# The default is "4,6" and not "2,4,6" because the shallow rungs SATURATE: once
# the champion beats them almost every game they answer a question already
# settled, and an anchor that is saturated still costs a full match -- every one
# of those games pays for the CHAMPION's search. `random` and `greedy` stay out
# for the same reason; they can be put back with DAMA_AB_DEPTHS and
# DAMA_STRENGTH_BASELINES.
#
# The cost per move grows steeply with the depth -- each extra pair of plies is
# worth several times the previous one -- which is what dictates the cadence
# below.
AB_DEPTHS      = [int(d) for d in os.environ.get("DAMA_AB_DEPTHS", "4,6").split(",") if d.strip()]
# Fixed baseline opponents. "random,greedy" to put them back; empty = none.
STRENGTH_BASELINES = [s.strip() for s in
                      os.environ.get("DAMA_STRENGTH_BASELINES", "").split(",") if s.strip()]
# DEEP anchor, on its own cadence. Depth 8 costs 6.6 times depth 6: even 20
# games are ~24 minutes, i.e. five whole cycles. As a routine measurement it is
# unaffordable, but it is the only anchor almost certainly NOT saturated yet --
# at depth 6 the champion was already at 0.75-0.88 -- so it serves as a rare,
# well-spaced reference. 0 = off.
AB_DEEP_DEPTH  = int(os.environ.get("DAMA_AB_DEEP_DEPTH", "8"))
AB_DEEP_EVERY  = int(os.environ.get("DAMA_AB_DEEP_EVERY", "60"))
AB_DEEP_GAMES  = int(os.environ.get("DAMA_AB_DEEP_GAMES", "20"))
MCTS_BATCH     = int(os.environ.get("DAMA_MCTS_BATCH", "8"))       # leaf batching
USE_SE         = os.environ.get("DAMA_USE_SE", "1") != "0"         # squeeze-and-excitation blocks
# Number of workers. The default is CAPPED AT 64 even on machines with many more
# cores: beyond that threshold performance gets WORSE, because every worker is a
# process that imports torch and contends on the inference server's queue, and
# the coordination costs more than it gives. Without the cap, a machine with
# hundreds of cores would start hundreds of workers, far past the point where
# adding them stops paying.
_WORKERS_DEFAULT = min(max(1, (os.cpu_count() or 2) - 1), 64)
WORKERS        = int(os.environ.get("DAMA_WORKERS", str(_WORKERS_DEFAULT)))
DEVICE         = os.environ.get("DAMA_DEVICE", "cpu")
WEIGHTS_DIR    = os.environ.get("DAMA_WEIGHTS_DIR", "weights")
# --- C++ engine for self-play and arena (optional) ---------------------------
# Path of the dama_engine executable. Empty = disabled, the Python path is used.
# The C++ engine replaces the self-play and arena games: training and the
# promotion decision stay in PyTorch/Python.
CPP_ENGINE     = os.environ.get("DAMA_CPP_ENGINE", "")
CPP_THREADS    = int(os.environ.get("DAMA_CPP_THREADS", str(os.cpu_count() or 8)))
CPP_FP16       = os.environ.get("DAMA_CPP_FP16", "0") != "0"
# Dry run: runs the engine WITHOUT a network (internal fake backend), to check
# that the connection works -- export, invocation, dataset, reading back,
# training -- on a machine without LibTorch. The samples produced have NO
# playing value: do not use it for a real run.
CPP_FAKE       = os.environ.get("DAMA_CPP_FAKE", "0") != "0"
# Central inference server (inference_server.py): one process owns the GPU and
# batches the requests of all the workers. "auto" = on when the device is not
# the CPU (that is where it helps); "1"/"0" to force it. Forcing it to 1 on CPU
# is useful to TEST the path without a GPU.
INFER_SERVER   = os.environ.get("DAMA_INFERENCE_SERVER", "auto")
SERVER_BATCH   = int(os.environ.get("DAMA_SERVER_BATCH", "256"))
# How long the server WAITS to merge requests, in seconds. 0 = drain the queue
# and go at once.
#
# With many workers it is not needed: the queue is always full and batches form
# on their own. With FEW workers instead the server runs a forward pass per
# handful of positions -- four processes asking for eight leaves each give
# batches of 32 against a cap of 256, i.e. the card works at an eighth of its
# batching capacity. A wait of a few milliseconds lets the latecomers arrive and
# can be worth much more than it costs.
#
# It is NOT tuned: it depends on the machine, and it was chosen without a card
# to measure it on. It is found with benchmark.py --sweep, which tries the
# combinations on the spot. It stays 0 until someone measures, because a made-up
# value here would slow down the machines where the queue is already full.
SERVER_WAIT    = float(os.environ.get("DAMA_SERVER_WAIT", "0"))

# Games per SPRT block. It must stay HERE, after CPP_THREADS/WORKERS, because the
# right value depends on how many games the machine can play in parallel.
#
# The engine plays a whole block in parallel, so a block smaller than the number
# of threads leaves the rest of the machine idle: with a fixed block of 10 on a
# machine that can run hundreds of games at once, almost all of it waits. Below
# the number of threads the cost per game explodes; above it, it flattens, and a
# large block takes almost the same wall time as a small one.
#
# Larger blocks make the SPRT slightly less efficient in NUMBER of games (the
# result is checked less often, so the threshold is overshot a little before
# stopping), but the extra games are almost free while extra blocks really cost.
# The trade-off is clear.
#
# Capped at 100: beyond it, with SPRT_MAX_GAMES=400, there would be fewer than
# four chances to stop early and the test would in effect become fixed-sample,
# losing the advantage of being sequential.
_PARALLELISM   = CPP_THREADS if CPP_ENGINE else WORKERS
SPRT_CHUNK     = int(os.environ.get(
    "DAMA_SPRT_CHUNK", str(max(10, min(100, _PARALLELISM)))))


def new_net():
    return PolicyValueNet(in_planes=IN_PLANES, channels=CHANNELS,
                                  n_blocks=N_BLOCKS, use_se=USE_SE)


def play_match(eval_a, eval_b, n_games: int, n_sims: int, seed: int = 0):
    """A vs B, both through greedy MCTS (no noise). They alternate White.
    Returns A's (wins, draws, losses) -- counts, not a mean, because the SPRT
    needs the spread of the outcomes."""
    from mcts import MCTS
    w = d = l = 0
    for g in range(n_games):
        a_is_white = (g % 2 == 0)
        pos = Position()
        rng = np.random.default_rng(seed + g)
        plies = 0
        while not pos.is_terminal() and plies < 300:
            ev = eval_a if (pos.turn == 1) == a_is_white else eval_b
            root = MCTS(ev, n_sims=n_sims, batch_size=MCTS_BATCH, rng=rng).run(pos, add_noise=False)
            move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            pos = pos.play(move)
            plies += 1
        r = pos.result() or 0
        a_res = r if a_is_white else -r
        if a_res > 0: w += 1
        elif a_res == 0: d += 1
        else: l += 1
    return w, d, l


def strength_vs_baseline(champion_eval, baseline, n_games: int, n_sims: int,
                         seed: int = 0) -> float:
    """Win rate of the champion (greedy MCTS) against a fixed-strength opponent
    (which picks its move directly). They alternate White. Score in [0,1]."""
    from mcts import MCTS
    score = 0.0
    for g in range(n_games):
        champ_white = (g % 2 == 0)
        pos = Position()
        rng = np.random.default_rng(seed + g)
        plies = 0
        while not pos.is_terminal() and plies < 300:
            champ_turn = (pos.turn == 1) == champ_white
            if champ_turn:
                root = MCTS(champion_eval, n_sims=n_sims, batch_size=MCTS_BATCH, rng=rng).run(pos, add_noise=False)
                move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            else:
                move = baseline.move(pos, rng)
            pos = pos.play(move)
            plies += 1
        r = pos.result() or 0
        champ_res = r if champ_white else -r
        score += 1.0 if champ_res > 0 else (0.5 if champ_res == 0 else 0.0)
    return score / n_games


def main():
    global PARALLEL
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.manual_seed(42)

    # Temporary folder of the C++ engine, ONE PER PROCESS.
    #
    # A fixed shared path (weights/cpp_tmp) with fixed names inside --
    # champion_jit.pt, selfplay.bin, arena_a.pt, arena_b.pt -- let two runs on
    # the same folder export and read back the same files, and the arena could
    # end up comparing the other run's networks. With the PID in the name this
    # cannot happen even if the lock were bypassed.
    #
    # Keeping it under WEIGHTS_DIR leaves it on the same disk as the weights (the
    # exported models weigh a few MB and pass through here every cycle) and keeps
    # it visible, instead of vanishing into a system folder.
    cpp_workdir = os.path.join(WEIGHTS_DIR, f"cpp_tmp_{os.getpid()}")
    os.makedirs(cpp_workdir, exist_ok=True)
    # Leftovers of dead runs: we hold the exclusive lock, so any other cpp_tmp_*
    # necessarily belongs to a process that no longer exists.
    for stale in glob.glob(os.path.join(WEIGHTS_DIR, "cpp_tmp*")):
        if os.path.abspath(stale) != os.path.abspath(cpp_workdir):
            shutil.rmtree(stale, ignore_errors=True)
            print(f"[dama] removed a temporary folder of an earlier run: {stale}")

    champion = new_net().to(DEVICE)
    ckpt = os.path.join(WEIGHTS_DIR, "champion.pt")
    if os.path.exists(ckpt):
        try:
            champion.load_state_dict(
                torch.load(ckpt, map_location=DEVICE, weights_only=True))
            print(f"[dama] resumed the champion from {ckpt}")
        except Exception as e:
            print(f"[dama] cannot resume {ckpt} ({e}); starting from scratch")
    buffer = deque(maxlen=BUFFER_SAMPLES)
    state_path = os.path.join(WEIGHTS_DIR, "train_state.pkl")
    start_cycle = 1
    if os.path.exists(state_path):
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            buffer.extend(state["buffer"])
            start_cycle = state["cycle"] + 1
            print(f"[dama] resumed the state from {state_path}: "
                  f"cycle {state['cycle']} completed, buffer={len(buffer)} samples")
        except Exception as e:
            print(f"[dama] cannot resume {state_path} ({e}); restarting from cycle 1")
    _by_name = {"random": RandomPlayer, "greedy": GreedyPlayer}
    baselines = ([_by_name[n]() for n in STRENGTH_BASELINES if n in _by_name]
                 + [AlphaBetaPlayer(d) for d in AB_DEPTHS])
    csv_log = MetricsCsv(WEIGHTS_DIR)
    net_kwargs = dict(channels=CHANNELS, n_blocks=N_BLOCKS,
                      in_planes=IN_PLANES, use_se=USE_SE)

    # Persistent worker pool for the whole run (no respawn per cycle).
    #
    # Two ways to parallelize, depending on where the network runs:
    #   - CPU  : each worker has its own copy of the network (no server).
    #   - GPU  : the workers stay CPU processes (move generation/MCTS/encoding)
    #            and a dedicated SERVER owns the GPU, merging their requests into
    #            large batches. Without it, on GPU self-play ran as a single
    #            process and the card stayed almost idle.
    use_server = (INFER_SERVER == "1" or
                  (INFER_SERVER == "auto" and DEVICE != "cpu"))
    PARALLEL = WORKERS > 1 and (DEVICE == "cpu" or use_server)
    cand_ckpt = os.path.join(WEIGHTS_DIR, "candidate.pt")
    pool = None
    server = None

    # With the C++ engine on, the Python pool and the inference server are NOT
    # needed: the C++ covers both self-play and arena. Creating them anyway meant
    # keeping WORKERS processes that import torch (several tens of GB of RAM with
    # 64 workers) plus a second CUDA context, all idle for the whole run -- and
    # each one is something that can die.
    #
    # They are still needed as a FALLBACK. So they are created at the first real
    # need (see ensure_pool): the normal run pays nothing and the safety net stays
    # available.
    lazy_parallel = PARALLEL and bool(CPP_ENGINE)
    if lazy_parallel:
        PARALLEL = False
        torch.save(champion.state_dict(), ckpt)   # needed by the C++ engine anyway
        print("[dama] Python pool and inference server NOT started: "
              "the C++ engine covers self-play and arena "
              "(they start only on a fallback)")

    if PARALLEL:
        torch.save(champion.state_dict(), ckpt)   # the workers read the champion from the file
        if use_server:
            server = InferenceServer(net_kwargs, DEVICE, WORKERS,
                                     max_batch=SERVER_BATCH,
                                     poll_timeout=SERVER_WAIT)
            print(f"[dama] inference server running on {DEVICE} "
                  f"(max batch {SERVER_BATCH}, {WORKERS} CPU workers)")
        pool = WorkerPool(net_kwargs, WORKERS, server=server)
        # Each worker already limits itself to 1 thread (parallel.py), but the
        # main process does not: by default PyTorch would use up to all the
        # cores for its own training, WHILE the pool of 15 workers stays alive
        # and waiting (persistent). Found with faulthandler: training crashed
        # with a native "access violation" inside the network's forward pass
        # under this contention -- oversubscription/conflict in the BLAS/MKL
        # library between the main process and the idle workers, not a
        # deterministic bug in the Python code. Same limit as the workers here.
        torch.set_num_threads(1)

    if CPP_ENGINE:
        print(f"[dama] self-play and arena from the C++ engine: {CPP_ENGINE} "
              f"({CPP_THREADS} threads, fp16={'yes' if CPP_FP16 else 'no'}); "
              "training stays in Python")
        if CPP_FAKE:
            print("[dama] WARNING: DAMA_CPP_FAKE is on -> engine without a network, "
                  "the samples have NO playing value (connection test only)")

    def ensure_pool():
        """Starts the pool and the server ONLY at the first need (fallback from
        the C++ engine).

        On the normal path with the C++ engine they are never created: no
        WORKERS idle processes with torch loaded, no second CUDA context, no
        processes that can die during a run of hours."""
        nonlocal pool, server
        if pool is not None or not lazy_parallel:
            return pool
        if use_server:
            server = InferenceServer(net_kwargs, DEVICE, WORKERS,
                                     max_batch=SERVER_BATCH,
                                     poll_timeout=SERVER_WAIT)
            print(f"[dama] (fallback) inference server started on {DEVICE}")
        pool = WorkerPool(net_kwargs, WORKERS, server=server)
        torch.set_num_threads(1)
        print(f"[dama] (fallback) Python pool started with {WORKERS} workers")
        return pool

    def run_match(net_a, net_b, n_games, seed, pool_, path_a, path_b,
                  reuse_jit=False):
        """Plays net_a against net_b and returns A's (wins, draws, losses).

        The SINGLE dispatch point between the three possible paths (C++ engine,
        worker pool, sequential), used both by the promotion arena and by the
        generation ladder: the three-way logic exists once, so the two cannot
        diverge.

        Counts and not the mean, because the SPRT needs the spread of the
        outcomes.
        """
        if CPP_ENGINE:
            try:
                _sc, wdl, out = cpp_engine.arena(
                    CPP_ENGINE,
                    None if CPP_FAKE else net_a,
                    None if CPP_FAKE else net_b,
                    n_games=n_games, n_sims=N_SIMS, n_threads=CPP_THREADS,
                    device=DEVICE, fp16=CPP_FP16, mcts_batch=MCTS_BATCH,
                    max_batch=SERVER_BATCH, seed=seed, in_planes=IN_PLANES,
                    c_puct=C_PUCT, temp_plies=ARENA_TEMP_PLIES,
                    workdir=cpp_workdir,
                    reuse_jit=reuse_jit)
                # If the engine reports how many games are really different and
                # they are far fewer than those played, the SPRT is counting
                # repetitions as independent evidence: it must be said at once,
                # because the downstream symptom -- promotions that do not hold
                # up in the generational comparison -- arrives many cycles later.
                dist = cpp_engine.parse_arena_distinct(out)
                if dist is not None and dist * 2 < n_games:
                    print(f"[dama] WARNING arena: only {dist} different games "
                          f"out of {n_games}. Raise DAMA_ARENA_TEMP_PLIES: the SPRT "
                          f"is treating repetitions as new evidence.")

                # The batches of the two hubs, once per run. The arena is two
                # thirds of the cycle and its bottleneck is precisely batching:
                # two different networks cannot share a batch, so each hub sees
                # about half of them. Without this line, on a new machine there
                # is no way to notice that the batches are small other than
                # running the benchmark again. It is tuned with benchmark.py
                # --sweep-arena.
                if not getattr(run_match, "_batches_reported", False):
                    batches = cpp_engine.parse_hub_batches(out)
                    if batches:
                        run_match._batches_reported = True
                        print("[dama][cpp] arena batches: "
                              + ", ".join(f"{x:.1f}" for x in batches)
                              + " positions per hub")
                return wdl
            except cpp_engine.CppEngineError as e:
                print(f"[dama] C++ match failed ({e}); Python path")
                pool_ = ensure_pool() or pool_

        if pool_ is not None:
            torch.save(net_a.state_dict(), path_a)
            torch.save(net_b.state_dict(), path_b)
            return pool_.arena(path_a, path_b, n_games, N_SIMS, MCTS_BATCH, seed)

        return play_match(NetEvaluator(net_a, DEVICE), NetEvaluator(net_b, DEVICE),
                          n_games, n_sims=N_SIMS, seed=seed)

    # GENERATIONAL meter: generation 0 is the starting network, with Elo 0 by
    # definition. See ladder.py for why it is needed.
    gen_ladder = Ladder(WEIGHTS_DIR)
    if GEN_EVERY > 0 and gen_ladder.is_empty():
        gen_ladder.add(0, 0.0, lambda p: torch.save(champion.state_dict(), p))
        print("[dama] ladder: generation 0 recorded (reference Elo 0)")

    # The anti-drift guard NEEDS the generation ladder: it takes its frozen
    # reference from there. With the ladder off it would never find an anchor and
    # would do nothing -- while staying on, i.e. giving the impression of
    # protecting.
    #
    # A protection that cannot work must fail at startup, not stay silent for a
    # hundred cycles.
    if ANTIDRIFT > 0:
        if GEN_EVERY <= 0:
            raise SystemExit(
                "[dama] DAMA_ANTIDRIFT is on but DAMA_GEN_EVERY is 0.\n"
                "       The guard takes its reference from the generations frozen\n"
                "       by the ladder: without the ladder it would protect nothing,\n"
                "       silently. Turn the ladder on (DAMA_GEN_EVERY=10) or\n"
                "       turn the guard off (DAMA_ANTIDRIFT=0).")
        print(f"[dama] anti-drift guard: {ANTIDRIFT} games against the "
              f"generation {ANTIDRIFT_BACK} back, threshold {ANTIDRIFT_MIN}")
        print(f"[dama] the guard starts protecting from cycle "
              f"{GEN_EVERY * ANTIDRIFT_BACK}: before that there are not enough "
              f"frozen generations")

    print(f"[dama] start: cycles={CYCLES} games/cycle={GAMES_PER_CYCLE} "
          f"sims={N_SIMS} net={CHANNELS}x{N_BLOCKS} device={DEVICE}")

    for cycle in range(start_cycle, CYCLES + 1):
        t0 = time.time()
        champ_eval = None if PARALLEL else NetEvaluator(champion, DEVICE)

        # Progress through the run, in [0,1]: it drives both the schedule of the
        # self-play simulations and the learning-rate decay. Being a function of
        # the cycle number only, it survives resumes.
        prog = (cycle - 1) / max(1, CYCLES - 1)
        sims_sp = int(round(N_SIMS + (SIMS_FINAL - N_SIMS) * prog))

        # 1) SELF-PLAY with the champion. Three possible paths, in order of
        #    preference: C++ engine (if enabled), Python worker pool, sequential.
        #    Training always stays in Python; the arena (run_match) makes the
        #    same choice between the engine and the Python paths.
        n_samples, results, game_stats = 0, [], []
        t_phase = time.time()
        cpp_done = False
        cpp_stats = {}
        if CPP_ENGINE:
            try:
                samples, engine_out = cpp_engine.selfplay(
                    CPP_ENGINE, None if CPP_FAKE else champion,
                    n_games=GAMES_PER_CYCLE, n_sims=sims_sp,
                    n_threads=CPP_THREADS, device=DEVICE, fp16=CPP_FP16,
                    mcts_batch=MCTS_BATCH, max_batch=SERVER_BATCH,
                    seed=cycle * 100000, in_planes=IN_PLANES, c_puct=C_PUCT,
                    value_discount=VALUE_DISCOUNT, workdir=cpp_workdir)
                buffer.extend(samples)
                n_samples += len(samples)
                cpp_done = True
                cpp_stats = cpp_engine.parse_stats(engine_out)
                if cycle == start_cycle:      # once only: visual confirmation
                    for line in engine_out.strip().splitlines():
                        if "games/hour" in line or "hub batches" in line:
                            print(f"[dama][cpp] {line.strip()}")
            except cpp_engine.CppEngineError as e:
                # Fall back on the Python path instead of stopping the run: a
                # long run must not die because of a problem in the external
                # engine.
                print(f"[dama] C++ engine failed ({e}); "
                      "self-play of this cycle on the Python path")
                if ensure_pool() is not None:
                    PARALLEL = True

        if cpp_done:
            pass
        elif PARALLEL:
            games = pool.selfplay(ckpt, GAMES_PER_CYCLE, sims_sp, MCTS_BATCH,
                                  cycle * 100000, c_puct=C_PUCT,
                                  value_discount=VALUE_DISCOUNT)
            # Failed games must be COUNTED, not just skipped. Discarding them
            # silently means a run can lose half of its data every cycle while
            # showing only a lower sample count -- which varies from cycle to
            # cycle anyway, so it alarms nobody. It is the defect noticed after
            # days, or never.
            failed = sum(1 for r in games if r is None)
            if failed:
                print(f"[dama] WARNING: {failed} games out of {len(games)} "
                      f"({100*failed/max(1,len(games)):.0f}%) failed in the "
                      f"worker processes and were discarded. "
                      f"The cause is printed by the processes themselves.")
            for res in games:
                if res is None:              # game failed in the worker, discarded
                    continue
                samples, z_white, plies, gstat = res
                buffer.extend(samples)
                n_samples += len(samples)
                results.append(z_white)
                game_stats.append(gstat)
        else:
            for g in range(GAMES_PER_CYCLE):
                samples, z_white, plies, gstat = play_game(
                    champ_eval, n_sims=sims_sp, batch_size=MCTS_BATCH, c_puct=C_PUCT,
                    seed=cycle * 100000 + g, value_discount=VALUE_DISCOUNT)
                buffer.extend(samples)
                n_samples += len(samples)
                results.append(z_white)
                game_stats.append(gstat)
        t_selfplay = time.time() - t_phase

        # 2) TRAIN a copy of the champion
        t_phase = time.time()
        candidate = new_net().to(DEVICE)
        candidate.load_state_dict(champion.state_dict())
        # Learning rate with cosine decay over the cycle number: high at the start
        # (the network is far off), low at the end (it settles instead of
        # oscillating).
        lr_cycle = LR * (LR_FINAL_FRAC + (1.0 - LR_FINAL_FRAC) *
                         0.5 * (1.0 + math.cos(math.pi * prog)))
        metrics = train_on_samples(candidate, list(buffer), epochs=EPOCHS,
                                   batch_size=BATCH_SIZE, lr=lr_cycle,
                                   device=DEVICE, use_ema=USE_EMA,
                                   warmup_frac=0.05 if LR_SCHEDULE else 0.0,
                                   lr_final_frac=0.1 if LR_SCHEDULE else 1.0)
        t_train = time.time() - t_phase

        # 3) ARENA: promotion if the candidate beats the champion.
        t_phase = time.time()
        if USE_SPRT:
            # Play in blocks and ask the SPRT after each one: clear cases close
            # early, uncertain ones get more evidence up to the cap. The seed
            # changes per block, otherwise every block would replay the very
            # same games.
            w = d = l = 0
            while w + d + l < SPRT_MAX_GAMES:
                chunk = min(SPRT_CHUNK, SPRT_MAX_GAMES - (w + d + l))
                # The two networks do not change for the whole match: they are
                # exported to TorchScript only at the first block. The SPRT calls
                # this line once per block, several times per cycle, and
                # re-exporting both networks at every call would pay again for
                # work already done.
                cw, cd, cl = run_match(candidate, champion, chunk,
                                       cycle * 7919 + (w + d + l),
                                       pool, cand_ckpt, ckpt,
                                       reuse_jit=(w + d + l) > 0)
                w += cw; d += cd; l += cl
                if sprt.evaluate(w, d, l, alpha=SPRT_ALPHA,
                                 beta=SPRT_BETA).decision != "continue":
                    break
            promoted = sprt.decide(w, d, l, PROMOTE_MIN,
                                   alpha=SPRT_ALPHA, beta=SPRT_BETA)
        else:
            w, d, l = run_match(candidate, champion, ARENA_GAMES, cycle * 7919,
                                pool, cand_ckpt, ckpt)
            promoted = (w + 0.5 * d) / max(1, w + d + l) >= PROMOTE_MIN
        t_arena = time.time() - t_phase
        arena_games_played = w + d + l
        score = (w + 0.5 * d) / max(1, arena_games_played)

        # 3b) GUARD AGAINST DRIFT (optional, DAMA_ANTIDRIFT=0 turns it off).
        #
        # The arena compares the candidate with the CURRENT champion, so it only
        # guarantees that it is better than the immediately previous link. A
        # chain of steps each just above parity can wander instead of climbing,
        # because the yardstick moves with the player -- and the generation
        # ladder, being relative too, cannot see it happening.
        #
        # The remedy is a yardstick that does NOT move: the candidate must also
        # hold up against an OLD, frozen generation. A random walk often beats its
        # own previous step, but does not keep beating a fixed reference.
        antidrift = ""
        if promoted and ANTIDRIFT > 0:
            drift_anchors = gen_ladder.anchors(ANTIDRIFT_BACK)
            if len(drift_anchors) >= ANTIDRIFT_BACK:
                old_gen = drift_anchors[-1]     # the farthest among those kept
                ref = new_net().to(DEVICE)
                ref.load_state_dict(torch.load(
                    os.path.join(WEIGHTS_DIR, old_gen["file"]), map_location=DEVICE))
                ref_ckpt = os.path.join(cpp_workdir, "antidrift.pt")
                torch.save(ref.state_dict(), ref_ckpt)
                wv, dv, lv = run_match(candidate, ref, ANTIDRIFT,
                                       cycle * 6271, pool, cand_ckpt, ref_ckpt)
                sv = (wv + 0.5 * dv) / max(1, wv + dv + lv)
                antidrift = f"{sv:.2f}vs_gen{old_gen['gen']}"
                if sv < ANTIDRIFT_MIN:
                    promoted = False
                    print(f"[dama] promotion BLOCKED by the anti-drift guard: against "
                          f"generation {old_gen['gen']} (cycle {old_gen['cycle']}) "
                          f"the candidate scores {sv:.3f} < {ANTIDRIFT_MIN}. "
                          f"The arena said {score:.3f}.")

        if promoted:
            champion = candidate

        # 4) save + log. In parallel mode champion.pt is updated ONLY if it
        #    changes (the workers reload it at the new mtime); in sequential mode
        #    it is always saved (for resuming).
        if promoted or not PARALLEL:
            torch.save(champion.state_dict(), ckpt)
        dt = time.time() - t0
        print(f"[cycle {cycle:03d}] samples={n_samples} buffer={len(buffer)} "
              f"loss={metrics.get('loss', 0):.3f} (p={metrics.get('policy', 0):.3f} "
              f"v={metrics.get('value', 0):.3f})  "
              f"arena={score:.2f}({arena_games_played}g) "
              + (f"[guard {antidrift}] " if antidrift else "")
              + f"{'PROMOTED' if promoted else 'rejected'}  {dt:.1f}s")

        # 5) STRENGTH TEST against fixed opponents
        t_phase = time.time()
        strength_str = ""
        # The deep anchor has its own cadence: it costs 6.6 times depth 6 and
        # cannot be part of the ordinary round (see AB_DEEP_EVERY). When it is
        # due, it is added to the same measurement, so it ends up in the same
        # CSV row and stays comparable.
        deep_due = (AB_DEEP_DEPTH > 0 and AB_DEEP_EVERY > 0
                    and cycle % AB_DEEP_EVERY == 0)
        if (STRENGTH_EVERY > 0 and cycle % STRENGTH_EVERY == 0) or deep_due:
            champ_eval = NetEvaluator(champion, DEVICE)
            todo = [(bl, STRENGTH_GAMES) for bl in baselines] if (
                STRENGTH_EVERY > 0 and cycle % STRENGTH_EVERY == 0) else []
            if deep_due:
                todo.append((AlphaBetaPlayer(AB_DEEP_DEPTH), AB_DEEP_GAMES))
            parts, csv_parts = [], []
            for bl, n_g in todo:
                wr = strength_vs_baseline(champ_eval, bl, n_g,
                                          n_sims=N_SIMS, seed=cycle * 104729)
                parts.append(f"vs {bl.name}={wr:.2f}({n_g}g)")
                csv_parts.append(f"{bl.name}={wr:.3f}")
            strength_str = ";".join(csv_parts)
            print(f"[strength {cycle:03d}] " + "  ".join(parts))

        # 5b) GENERATIONAL MEASUREMENT: the current champion against the past
        #     champions. Unlike step 5 it does NOT saturate, because the
        #     yardstick grows with the player.
        elo_str = ""
        if GEN_EVERY > 0 and cycle % GEN_EVERY == 0:
            anchors = gen_ladder.anchors(GEN_ANCHORS)
            if anchors:
                ref_net = new_net().to(DEVICE)
                scores = []
                for a in anchors:
                    try:
                        ref_net.load_state_dict(
                            torch.load(gen_ladder.path_of(a), map_location=DEVICE,
                                       weights_only=True))
                    except Exception as e:
                        print(f"[dama] anchor gen{a['gen']} unreadable ({e}); skipped")
                        continue
                    gw, gd, gl = run_match(
                        champion, ref_net, GEN_GAMES,
                        cycle * 31337 + a["gen"], pool,
                        os.path.join(WEIGHTS_DIR, "ladder_a.pt"),
                        os.path.join(WEIGHTS_DIR, "ladder_b.pt"))
                    scores.append((a, (gw + 0.5 * gd) / max(1, gw + gd + gl)))
                if scores:
                    elo, details = gen_ladder.estimate_elo(scores)
                    prev = gen_ladder.latest_elo()
                    entry = gen_ladder.add(
                        cycle, elo,
                        lambda p: torch.save(champion.state_dict(), p))
                    print(f"[elo      {cycle:03d}] gen{entry['gen']} "
                          f"Elo={elo:+.0f} (delta {elo - prev:+.0f})  "
                          + "  ".join(details))
                    elo_str = f"{elo:.1f}"
        t_eval = time.time() - t_phase

        # 5c) METRICS to CSV: one row per cycle, which is what makes an end-of-run
        #     plot possible.
        # Python path: the per-game statistics are aggregated. C++ path: the
        # engine already aggregated and printed them, and they are read back.
        gs = aggregate(game_stats)
        if cpp_stats:
            gs.games = int(cpp_stats.get("games", 0))
            gs.plies_mean = cpp_stats.get("plies_mean", 0.0)
            gs.plies_max = int(cpp_stats.get("plies_max", 0))
            gs.white_wins = int(cpp_stats.get("white", 0))
            gs.black_wins = int(cpp_stats.get("black", 0))
            gs.draws = int(cpp_stats.get("draws", 0))
            gs.truncated = int(cpp_stats.get("truncated", 0))
            gs.no_progress = int(cpp_stats.get("no_progress", 0))
            gs.captures_mean = cpp_stats.get("captures_mean", 0.0)
            gs.promotions_mean = cpp_stats.get("promotions_mean", 0.0)
            gs.entropy_mean = cpp_stats.get("entropy_mean", 0.0)
            gs.q_spread_mean = cpp_stats.get("q_spread_mean", 0.0)
            gs.value_mae = cpp_stats.get("value_mae", 0.0)
        # duration_s: the REAL cycle time, read here and not earlier.
        #
        # Recording `dt`, timed right after the arena and therefore BEFORE the
        # evaluation phase, leaves out the whole evaluation on the cycles that
        # have one: that time appears in no column, and every end-of-run estimate
        # built on this one comes out short.
        #
        # The number printed on screen stays the pre-evaluation one: that line is
        # printed before the evaluation starts, so it could not include it. The
        # [strength]/[elo] lines that follow make it clear in the log that more
        # happened afterwards.
        csv_log.append(CycleRow(
            cycle=cycle, duration_s=round(time.time() - t0, 2),
            samples=n_samples, buffer=len(buffer), sims=sims_sp,
            loss=round(metrics.get("loss", 0.0), 5),
            loss_policy=round(metrics.get("policy", 0.0), 5),
            loss_value=round(metrics.get("value", 0.0), 5),
            arena_score=round(score, 4), promoted=int(promoted),
            games=gs.games, plies_mean=round(gs.plies_mean, 2),
            plies_max=gs.plies_max, white_wins=gs.white_wins,
            black_wins=gs.black_wins, draws=gs.draws,
            draw_rate=round(gs.draw_rate, 4), truncated=gs.truncated,
            no_progress=gs.no_progress,
            captures_mean=round(gs.captures_mean, 3),
            promotions_mean=round(gs.promotions_mean, 3),
            entropy_mean=round(gs.entropy_mean, 4),
            q_spread_mean=round(gs.q_spread_mean, 4),
            value_mae=round(gs.value_mae, 4),
            elo=elo_str, strength=strength_str,
            t_selfplay=round(t_selfplay, 2), t_train=round(t_train, 2),
            t_arena=round(t_arena, 2), t_eval=round(t_eval, 2)))

        # 6) resume checkpoint: completed cycle + buffer, atomic write
        tmp_path = state_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump({"cycle": cycle, "buffer": list(buffer)}, f)
        os.replace(tmp_path, state_path)

    if pool is not None:
        pool.close()
    if server is not None:
        server.close()
    print("[dama] done.")


def _main_with_lock():
    """Starts the run holding the exclusive lock on the weights folder.

    The lock lives HERE and not in run.sh because run.sh can be bypassed: a run
    started with a bare `python conductor.py` while another one is alive would
    overwrite its weights, its state and its metrics. A check that lives only in
    the launch script protects only those who use the launch script.
    """
    # A --help that answers: these tools take no arguments, so without this
    # line they would start computing (see env_help.py).
    from env_help import help_if_requested
    help_if_requested(__doc__, __file__)
    try:
        with acquire(WEIGHTS_DIR):
            main()
    except RunLockError as e:
        print(f"\n[dama] ERROR: {e}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    _main_with_lock()
