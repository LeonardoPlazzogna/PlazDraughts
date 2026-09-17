"""
End-to-end test of MCTS (leaf batching) + self-play + training (compact masked
policy + WDL). Runnable: `python tests/test_alphazero.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from dama import Position, EMPTY, W_MAN, B_MAN, WHITE, POLICY_SIZE, IN_PLANES
from dama import encode, legal_mask
from mcts import MCTS
from evaluators import UniformEvaluator
from selfplay import play_game
from train import train_on_samples
from model import PolicyValueNet

FAIL = 0


def check(name, cond):
    global FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAIL += 1


def test_model_interface():
    net = PolicyValueNet(channels=32, n_blocks=2, in_planes=IN_PLANES).eval()
    p = Position()
    x = torch.from_numpy(encode(p)).unsqueeze(0)
    with torch.no_grad():
        pl, v, wdl = net(x)
    check("model: policy[1,POLICY_SIZE] value[1,1] wdl[1,3]",
          pl.shape == (1, POLICY_SIZE) and v.shape == (1, 1) and wdl.shape == (1, 3))
    mask = torch.from_numpy(legal_mask(p)).unsqueeze(0)
    probs = torch.softmax(pl.masked_fill(mask == 0, -1e9), dim=1)
    check("model: masked softmax = 1 on the legal moves, 0 on the illegal ones",
          abs(float(probs.sum()) - 1.0) < 1e-5 and float(probs[mask == 0].sum()) == 0.0)


def test_mcts_batched():
    p = Position()
    root = MCTS(UniformEvaluator(), n_sims=48, batch_size=8).run(p, add_noise=False)
    total = sum(c.N for c in root.children.values())
    check("mcts: children = legal moves (7)", len(root.children) == 7)
    check("mcts: total child visits = n_sims (correct batching)", total == 48)
    best = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    check("mcts: best move is legal", best in p.legal_moves())


def test_mcts_finds_win():
    b = [EMPTY] * 64
    b[42] = W_MAN
    b[33] = B_MAN          # 42 captures 33 -> Black is left without pieces
    p = Position(b, WHITE)
    root = MCTS(UniformEvaluator(), n_sims=32, batch_size=8).run(p, add_noise=False)
    move, child = next(iter(root.children.items()))
    check("mcts: the winning move is the capture", move.is_capture and 33 in move.captured)
    check("mcts: value ~ +1 on the winning move", -child.Q > 0.9)


def test_selfplay_samples():
    samples, z_white, plies, stats = play_game(UniformEvaluator(), n_sims=16,
                                               batch_size=8, seed=1)
    ok = all(s[0].shape == (IN_PLANES, 8, 8) and s[1].shape == (POLICY_SIZE,)
             and s[2].shape == (POLICY_SIZE,) for s in samples)
    ok_pi = all(abs(float(s[1].sum()) - 1.0) < 1e-4 for s in samples)
    ok_mask = all(float(s[2].sum()) >= 1.0 for s in samples)      # >=1 legal move
    ok_z = all(s[3] in (-1.0, 0.0, 1.0) for s in samples)
    check("selfplay: samples produced", len(samples) > 0)
    check("selfplay: shapes X[IN_PLANES,8,8], pi[POLICY_SIZE], mask[POLICY_SIZE]", ok)
    check("selfplay: pi sums to 1", ok_pi)
    check("selfplay: mask not empty", ok_mask)
    check("selfplay: z in {-1,0,1}", ok_z)

    # diagnostic statistics (metrics.py): they must be consistent with each other
    check("selfplay: stats.plies = returned plies", stats["plies"] == plies)
    check("selfplay: stats.z_white = returned z_white", stats["z_white"] == z_white)
    check("selfplay: entropy finite and non-negative",
          stats["entropy"] is not None and stats["entropy"] >= 0.0)
    check("selfplay: calibration error in [0,2]",
          stats["value_mae"] is not None and 0.0 <= stats["value_mae"] <= 2.0)
    check("selfplay: captures and promotions non-negative",
          stats["captures"] >= 0 and stats["promotions"] >= 0)
    check("selfplay: truncated and no-progress are mutually exclusive",
          not (stats["truncated"] and stats["no_progress"]))


def _loss(net, samples):
    net.eval()
    X = torch.from_numpy(np.stack([s[0] for s in samples]))
    P = torch.from_numpy(np.stack([s[1] for s in samples]))
    M = torch.from_numpy(np.stack([s[2] for s in samples]))
    Z = torch.from_numpy(np.array([s[3] for s in samples], np.float32)).unsqueeze(1)
    with torch.no_grad():
        pl, v, wdl = net(X)
        logp = F.log_softmax(pl.masked_fill(M == 0, -1e9), dim=1)
        ploss = -(P * logp).sum(1).mean()
        cls = ((Z <= 0).long() + (Z < 0).long()).squeeze(1)
        vloss = F.cross_entropy(wdl, cls)
    return float(ploss + vloss)


def test_training_learns():
    samples = []
    for g in range(3):
        s, _, _, _ = play_game(UniformEvaluator(), n_sims=12, batch_size=8, seed=10 + g)
        samples.extend(s)
    net = PolicyValueNet(channels=32, n_blocks=2, in_planes=IN_PLANES)
    before = _loss(net, samples)
    m = train_on_samples(net, samples, epochs=10, batch_size=64, lr=2e-3)
    after = _loss(net, samples)
    check("train: finite loss", np.isfinite(m["loss"]))
    check(f"train: the loss goes down ({before:.3f} -> {after:.3f})", after < before)


def main():
    for t in (test_model_interface, test_mcts_batched, test_mcts_finds_win,
              test_selfplay_samples, test_training_learns):
        print(f"--- {t.__name__} ---")
        t()
    print()
    if FAIL == 0:
        print(">>> ALL ALPHAZERO TESTS PASSED")
        sys.exit(0)
    print(f">>> {FAIL} TESTS FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
