"""
Evaluators for the MCTS: given a position they return (priors, value).
  priors : dict {Move -> probability} over the legal moves (sums to 1)
  value  : float in [-1, 1] from the point of view of the side to move

BATCH interface: `evaluate_batch(list[Position]) -> list[(priors, value)]`
evaluates many positions in ONE forward pass. It is the key to leaf batching:
it amortizes the per-call overhead (numpy -> torch, kernel launches) and makes
the GPU usable (at batch size 1 it sits idle).

The RAW forward pass (planes -> logits/values) and the construction of the
PRIORS are kept apart on purpose: the first is the only part that touches the
network (and on GPU it lives in the inference server process, see
inference_server.py), the second is plain numpy logic shared by the local and
the remote evaluator. So the two paths give the same priors BY CONSTRUCTION,
not by coincidence -- and that is tested (tests/test_inference_server.py).
"""
from __future__ import annotations
import numpy as np

from dama import legal_actions
from dama.encoder import encode


def priors_from_logits(pos, logits_row: np.ndarray):
    """Raw policy logits [POLICY_SIZE] -> {Move: probability} over the legal
    moves only (sums to 1).

    Index collisions (multiple captures sharing their first step): softmax over
    the DISTINCT indices only (the shared logit is not counted twice), then the
    probability of an index is split evenly among the moves sharing it. The
    model cannot tell them apart, so an even split is the principled choice,
    and the priors always sum to 1.
    """
    moves, indices = legal_actions(pos)
    if not moves:
        return {}
    uniq = list(dict.fromkeys(indices))                 # distinct, in order
    sel = logits_row[uniq].astype(np.float64)
    sel -= sel.max()                                    # numerically stable softmax
    exp = np.exp(sel)
    u_probs = exp / exp.sum()
    prob_of = {ix: float(p) for ix, p in zip(uniq, u_probs)}
    counts = {}
    for ix in indices:
        counts[ix] = counts.get(ix, 0) + 1
    return {m: prob_of[indices[j]] / counts[indices[j]]
            for j, m in enumerate(moves)}


class UniformEvaluator:
    def evaluate_batch(self, positions):
        return [self.evaluate(p) for p in positions]

    def evaluate(self, pos):
        moves = pos.legal_moves()
        if not moves:
            return {}, 0.0
        p = 1.0 / len(moves)
        return {m: p for m in moves}, 0.0


class NetEvaluator:
    """Network-based (torch) evaluator, with batched evaluation."""

    def __init__(self, net, device: str = "cpu"):
        import torch
        self.torch = torch
        net = net.to(device).eval()
        self.device = device
        # BatchNorm folded into the convolutions, plus inference fusions:
        # torch.jit.freeze inlines the weights as constants and folds the BN,
        # optimize_for_inference applies the fusions. Safe fallback: if anything
        # cannot be traced, the plain network is used (no change in behavior).
        # The trace uses a batch > 1 so the batch dimension stays dynamic (the
        # reshape uses x.size(0)).
        self.net = net
        try:
            ex = torch.zeros(4, net.in_planes, 8, 8, device=device)
            with torch.no_grad():
                m = torch.jit.trace(net, ex)
                m = torch.jit.freeze(m)
                m = torch.jit.optimize_for_inference(m)
                # check that it accepts batch sizes other than the traced one
                m(torch.zeros(1, net.in_planes, 8, 8, device=device))
            self.net = m
        except Exception:
            pass

    def raw_forward(self, X: np.ndarray):
        """Planes [B, in_planes, 8, 8] -> (policy_logits [B, POLICY_SIZE],
        values [B]), both numpy on CPU. This is the ONLY part that touches the
        network: on GPU it lives in the inference server process."""
        torch = self.torch
        x = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            policy_logits, value, _ = self.net(x)
        return (policy_logits.cpu().numpy(),
                value.cpu().reshape(-1).numpy())

    def evaluate(self, pos):
        return self.evaluate_batch([pos])[0]

    def evaluate_batch(self, positions):
        if not positions:
            return []
        X = np.stack([encode(p) for p in positions])
        logits, values = self.raw_forward(X)
        return [(priors_from_logits(pos, logits[i]), float(values[i]))
                for i, pos in enumerate(positions)]
