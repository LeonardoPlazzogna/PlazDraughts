"""
Multi-process PARALLEL self-play and arena, on a persistent worker pool.

Self-play in Python is GIL-bound: one game at a time uses ~1 core. Here N worker
processes play games in parallel, each with torch limited to **1 thread** (so
as not to oversubscribe: N processes x 1 thread = N cores). Without an inference
server each worker holds a CPU copy of the network; with one
(inference_server.py, typically on GPU) the workers delegate every evaluation to
the process that owns the device.

The functions are at module level because Windows spawn must be able to import
them (no closures).
"""
from __future__ import annotations
import multiprocessing as mp
import os


# 10 and not 4: see the longer comment in conductor.py. In short, 4 had never
# been verified, and measuring how many arena games are really distinct shows
# that it is not enough -- the repetitions enter the SPRT as if they were
# independent trials.
ARENA_TEMP_PLIES = int(os.environ.get("DAMA_ARENA_TEMP_PLIES", "10"))


def play_arena_game(eval_a, eval_b, n_sims, batch_size, a_is_white, seed,
                    temp_plies: int | None = None):
    """One arena game A vs B (they alternate White). Returns A's score:
    1.0 win, 0.5 draw, 0.0 loss.

    The first `temp_plies` plies are SAMPLED from the visits instead of played
    greedily. Without that the arena is fully deterministic -- no Dirichlet
    noise and a deterministic network -- and then every game with the same
    colors is THE SAME game: playing 30 of them gives exactly the same
    information as playing 2. Verified experimentally: four different seeds
    gave the very same result, and promotion (the gate that decides whether the
    network improves) in fact rested on 2 games. A few sampled plies diversify
    the openings without distorting the measurement, and they apply to both
    sides.
    """
    import numpy as np
    from mcts import MCTS
    from dama import Position
    from selfplay import select_move
    if temp_plies is None:
        temp_plies = ARENA_TEMP_PLIES
    pos = Position()
    rng = np.random.default_rng(seed)
    plies = 0
    while not pos.is_terminal() and plies < 300:
        ev = eval_a if (pos.turn == 1) == a_is_white else eval_b
        root = MCTS(ev, n_sims=n_sims, batch_size=batch_size, rng=rng).run(pos, add_noise=False)
        temp = 1.0 if plies < temp_plies else 0.0
        move = select_move(root, temp, rng)
        pos = pos.play(move)
        plies += 1
    r = pos.result() or 0
    a_res = r if a_is_white else -r
    return 1.0 if a_res > 0 else (0.5 if a_res == 0 else 0.0)


# ==================== PERSISTENT POOL ======================================
# The workers import torch ONCE (when the pool starts) and reload a network
# ONLY when its weights file changes (mtime). No respawn per cycle.
_PP = {}


def _pp_init(net_kwargs, server_qs=None, widx_counter=None):
    import torch
    torch.set_num_threads(1)
    _PP["kwargs"] = net_kwargs
    _PP["nets"] = {}          # path -> (mtime, NetEvaluator)
    _PP["server"] = None
    if server_qs is not None:
        # Inference-server mode (typically GPU): the worker loads no network
        # and does not touch the GPU; it talks to the server. Each worker takes
        # a UNIQUE index from a shared counter, which decides on which queue it
        # receives its own responses.
        req_q, resp_qs = server_qs
        with widx_counter.get_lock():
            widx = widx_counter.value
            widx_counter.value += 1
        _PP["server"] = (req_q, resp_qs[widx], widx)


def _pp_load(path):
    import os
    import torch
    from model import PolicyValueNet
    from evaluators import NetEvaluator
    if _PP.get("server") is not None:
        from inference_server import RemoteEvaluator
        req_q, resp_q, widx = _PP["server"]
        return RemoteEvaluator(path, req_q, resp_q, widx)
    mt = os.path.getmtime(path)
    cached = _PP["nets"].get(path)
    if cached is None or cached[0] != mt:
        net = PolicyValueNet(**_PP["kwargs"])
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        _PP["nets"][path] = (mt, NetEvaluator(net, "cpu"))
    return _PP["nets"][path][1]


# A failed game is discarded because the run must go on, but the CAUSE must be
# reported at least once. Silently swallowing the exception loses data without
# saying how much or why, and the only symptom is a lower sample count, which
# varies from cycle to cycle anyway.
#
# Once per process, not at every game: if the fault is systematic it would print
# one line per game and flood the log exactly when it needs to be read.
_already_reported = False


def _report_once(where: str, e: Exception) -> None:
    global _already_reported
    if not _already_reported:
        _already_reported = True
        print(f"[worker {os.getpid()}] game discarded in {where}: "
              f"{type(e).__name__}: {e}", flush=True)


def _pp_selfplay(task):
    path, seed, n_sims, batch, c_puct, discount = task
    from selfplay import play_game
    try:
        return play_game(_pp_load(path), n_sims=n_sims, batch_size=batch,
                         c_puct=c_puct, seed=seed, value_discount=discount)
    except Exception as e:
        _report_once("self-play", e)
        return None          # faulty game discarded, the run goes on


def _pp_arena(task):
    a_path, b_path, a_white, seed, n_sims, batch = task
    try:
        return play_arena_game(_pp_load(a_path), _pp_load(b_path), n_sims, batch, a_white, seed)
    except Exception as e:
        _report_once("arena", e)
        return None          # faulty game discarded, the run goes on


class WorkerPool:
    """Persistent worker pool for the whole run.

    With `server` (InferenceServer) the workers load no networks of their own
    and delegate every evaluation to the process that owns the GPU: this is the
    path that makes parallelism useful on GPU (see inference_server). Without
    `server`: one copy of the network per worker, on CPU.
    """

    def __init__(self, net_kwargs, n_workers: int, server=None):
        ctx = mp.get_context("spawn")
        if server is None:
            initargs = (net_kwargs, None, None)
        else:
            # shared counter: gives every worker a unique index
            initargs = (net_kwargs, server.queues(), ctx.Value("i", 0))
        self.pool = ctx.Pool(n_workers, initializer=_pp_init, initargs=initargs)

    def selfplay(self, champ_path, n_games, n_sims, batch, base_seed,
                 c_puct=1.5, value_discount=1.0):
        tasks = [(champ_path, base_seed + i, n_sims, batch, c_puct, value_discount)
                 for i in range(n_games)]
        return self.pool.map(_pp_selfplay, tasks, chunksize=1)

    def arena(self, a_path, b_path, n_games, n_sims, batch, base_seed):
        """Returns A's (wins, draws, losses).

        Counts rather than a mean: the SPRT (sprt.py) needs the SPREAD of the
        outcomes, not just the score -- with many draws the real variance is
        much lower than the binomial one, and using it shortens the matches for
        the same statistical guarantees.
        """
        tasks = [(a_path, b_path, i % 2 == 0, base_seed + i, n_sims, batch)
                 for i in range(n_games)]
        scores = [s for s in self.pool.map(_pp_arena, tasks, chunksize=1) if s is not None]
        w = sum(1 for s in scores if s > 0.75)
        d = sum(1 for s in scores if 0.25 <= s <= 0.75)
        l = sum(1 for s in scores if s < 0.25)
        return w, d, l                          # failed games are left out

    def close(self):
        self.pool.close()
        self.pool.join()
