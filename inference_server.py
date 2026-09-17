"""
Central inference server: unlocks parallel self-play on GPU.

PROBLEM. The worker pool (parallel.py) gives each process its own copy of the
network. That is fine on CPU, not on GPU: N processes that each open a CUDA
context waste VRAM and fight over the card, and every worker sends tiny batches
(8 positions) that leave a 4090 nearly idle. This is why `conductor.py` used to
turn parallelism off entirely on GPU -- with the result that the GPU worked for
a single process.

SOLUTION (the same architecture as the chess project). ONE process owns the GPU
and the network; N workers stay CPU processes that do move generation, MCTS and
encoding, and send it evaluation requests. The server merges the requests of
ALL the workers into one large batch and runs a single forward pass. So the
Python work (the real bottleneck) scales over the cores, and the GPU gets
batches large enough to be used for real.

    worker_1 ... worker_N   --(request queue)-->  SERVER (owns the GPU)
         ^                                              |
         +----------(response queues, one per worker)---+

CONTRACT. The server receives ONLY already-encoded planes and returns ONLY raw
logits and values: all the game logic (legal moves, mask, construction of the
priors with the even split of colliding indices) stays in the workers, shared
with the local evaluator through `evaluators.priors_from_logits`. The GPU path
and the CPU path therefore give the same priors by construction.

NOTE: this path was validated with the server on device "cpu", with no CUDA
available (the logic is identical, only where the forward pass runs changes).
Its performance on a 4090 was not measured.
"""
from __future__ import annotations
import multiprocessing as mp
import queue as _queue

import numpy as np

_STOP = "__stop__"


# ============================ SERVER process ============================

def _server_loop(req_q, resp_qs, net_kwargs, device, max_batch, poll_timeout):
    """Server process loop: merges requests and runs one forward pass per batch."""
    import os
    import torch
    from model import PolicyValueNet
    from evaluators import NetEvaluator

    nets = {}   # path -> (mtime, NetEvaluator)

    def get_net(path):
        mt = os.path.getmtime(path)
        cached = nets.get(path)
        if cached is None or cached[0] != mt:
            net = PolicyValueNet(**net_kwargs)
            net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            nets[path] = (mt, NetEvaluator(net, device))
        return nets[path][1]

    stopping = False
    while not stopping:
        try:
            first = req_q.get(timeout=1.0)
        except _queue.Empty:
            continue
        if first == _STOP:
            break

        batch = [first]
        # Coalescing: drain the queue up to max_batch. The more workers there
        # are, the larger the batch the GPU gets in one go.
        while len(batch) < max_batch:
            try:
                nxt = req_q.get(timeout=poll_timeout) if poll_timeout > 0 else req_q.get_nowait()
            except _queue.Empty:
                break
            if nxt == _STOP:
                stopping = True
                break
            batch.append(nxt)

        # Group by model: in the arena two different networks coexist in the
        # same wave, so one forward pass per distinct network is needed.
        by_path = {}
        for widx, path, X in batch:
            by_path.setdefault(path, []).append((widx, X))

        for path, items in by_path.items():
            try:
                ev = get_net(path)
                sizes = [X.shape[0] for _, X in items]
                big = np.concatenate([X for _, X in items], axis=0)
                logits, values = ev.raw_forward(big)
            except Exception as e:                    # isolated fault, sent to the requesters
                for widx, X in items:
                    resp_qs[widx].put(("error", repr(e)))
                continue
            off = 0
            for (widx, _X), n in zip(items, sizes):
                resp_qs[widx].put(("ok", logits[off:off + n].copy(),
                                   values[off:off + n].copy()))
                off += n


# ========================= WORKER side (client) =========================

class RemoteEvaluator:
    """Same interface as NetEvaluator, but the forward pass happens in the server.

    The workers never touch the GPU: they build the planes, send a request, wait
    for the logits and rebuild the priors locally with the same function the
    local evaluator uses.
    """

    # If the server dies, a get() without a timeout would leave the worker
    # hanging forever and the run would stall silently. With the timeout the
    # fault becomes an exception: the game is discarded by the workers'
    # try/except and the run goes on instead of freezing.
    RESPONSE_TIMEOUT = 300.0

    def __init__(self, path, req_q, resp_q, widx):
        self.path = path
        self.req_q = req_q
        self.resp_q = resp_q
        self.widx = widx

    def evaluate(self, pos):
        return self.evaluate_batch([pos])[0]

    def evaluate_batch(self, positions):
        if not positions:
            return []
        from dama.encoder import encode
        from evaluators import priors_from_logits

        X = np.stack([encode(p) for p in positions])
        self.req_q.put((self.widx, self.path, X))
        try:
            msg = self.resp_q.get(timeout=self.RESPONSE_TIMEOUT)
        except _queue.Empty:
            raise RuntimeError(
                f"no response from the inference server within "
                f"{self.RESPONSE_TIMEOUT:.0f}s (server dead?)")
        if msg[0] == "error":
            raise RuntimeError(f"inference server: {msg[1]}")
        _, logits, values = msg
        return [(priors_from_logits(pos, logits[i]), float(values[i]))
                for i, pos in enumerate(positions)]


# ====================== PARENT-process manager ======================

class InferenceServer:
    """Starts/stops the server process and hands out the queues to the workers."""

    def __init__(self, net_kwargs, device: str, n_workers: int,
                 max_batch: int = 256, poll_timeout: float = 0.0):
        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue()
        self.resp_qs = [ctx.Queue() for _ in range(n_workers)]
        self.proc = ctx.Process(
            target=_server_loop,
            args=(self.req_q, self.resp_qs, net_kwargs, device, max_batch, poll_timeout),
            daemon=True,
        )
        self.proc.start()

    def queues(self):
        """(request queue, response queues) to pass to the workers."""
        return self.req_q, self.resp_qs

    def close(self):
        try:
            self.req_q.put(_STOP)
            self.proc.join(timeout=10)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
