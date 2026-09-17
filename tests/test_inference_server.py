"""
Tests of the INFERENCE SERVER (inference_server.py).

The central check is PARITY: the evaluations obtained through the server must
coincide with those of the local evaluator on the same positions. The risk of
this architecture is not slowness but a self-play that plays in a subtly
different way without anyone noticing -- exactly as for the C++ port of the
engine.

Here the server runs on the "cpu" device, with no CUDA needed: the logic
tested is identical to the one that would run on a GPU, only where the forward
pass happens changes.
"""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dama import IN_PLANES, Position
from model import PolicyValueNet
from evaluators import NetEvaluator
from inference_server import InferenceServer, RemoteEvaluator

FAILS = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


def sample_positions(n=12):
    """Real positions reached by playing, not built by hand."""
    rng = np.random.default_rng(7)
    out, pos = [Position()], Position()
    while len(out) < n and not pos.is_terminal():
        moves = pos.legal_moves()
        pos = pos.play(moves[int(rng.integers(len(moves)))])
        out.append(pos)
    return out[:n]


def test_parity():
    print("--- test_parity (server vs local evaluator) ---")
    net = PolicyValueNet(channels=32, n_blocks=2, in_planes=IN_PLANES)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "net.pt")
    torch.save(net.state_dict(), path)

    # the local evaluator loads EXACTLY the same weights saved to file, so the
    # only difference between the two paths is where the forward pass happens
    net_local = PolicyValueNet(channels=32, n_blocks=2, in_planes=IN_PLANES)
    net_local.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    local = NetEvaluator(net_local, "cpu")

    net_kwargs = dict(channels=32, n_blocks=2, in_planes=IN_PLANES)
    server = InferenceServer(net_kwargs, "cpu", n_workers=1, max_batch=64)
    try:
        req_q, resp_qs = server.queues()
        remote = RemoteEvaluator(path, req_q, resp_qs[0], 0)

        positions = sample_positions(12)
        got_local = local.evaluate_batch(positions)
        got_remote = remote.evaluate_batch(positions)

        check("same number of results", len(got_local) == len(got_remote))

        max_prior_diff = 0.0
        max_value_diff = 0.0
        same_keys = True
        for (pl, vl), (pr, vr) in zip(got_local, got_remote):
            if set(pl.keys()) != set(pr.keys()):
                same_keys = False
                continue
            for m in pl:
                max_prior_diff = max(max_prior_diff, abs(pl[m] - pr[m]))
            max_value_diff = max(max_value_diff, abs(vl - vr))

        check("same legal moves in the priors", same_keys)
        print(f"      max |diff| priors = {max_prior_diff:.3e}")
        print(f"      max |diff| values = {max_value_diff:.3e}")
        check("identical priors (tol 1e-6)", max_prior_diff < 1e-6)
        check("identical values (tol 1e-6)", max_value_diff < 1e-6)

        # the priors are still a valid distribution
        ok_sum = all(abs(sum(pr.values()) - 1.0) < 1e-5
                     for pr, _ in got_remote if pr)
        check("priors from the server sum to 1", ok_sum)
    finally:
        server.close()


def test_batch_coalescing():
    """Requests of different sizes in the same wave must each go back to their
    own sender, with their own length."""
    print("--- test_batch_coalescing ---")
    net_kwargs = dict(channels=16, n_blocks=1, in_planes=IN_PLANES)
    net = PolicyValueNet(**net_kwargs)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "net.pt")
    torch.save(net.state_dict(), path)

    server = InferenceServer(net_kwargs, "cpu", n_workers=3, max_batch=128)
    try:
        req_q, resp_qs = server.queues()
        positions = sample_positions(10)
        sizes = [1, 4, 7]
        evs = [RemoteEvaluator(path, req_q, resp_qs[i], i) for i in range(3)]
        results = [ev.evaluate_batch(positions[:n]) for ev, n in zip(evs, sizes)]
        check("every request receives its own number of results",
              [len(r) for r in results] == sizes)
        check("no empty result",
              all(all(p for p, _ in r) for r in results))
    finally:
        server.close()


if __name__ == "__main__":
    test_parity()
    test_batch_coalescing()
    print()
    if FAILS:
        print(f">>> {len(FAILS)} TESTS FAILED: {FAILS}")
        sys.exit(1)
    print(">>> ALL INFERENCE SERVER TESTS PASSED")
