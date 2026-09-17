r"""
How much the network still has to learn from the data it is already generating.

THE QUESTION. A run that stalls can have stalled for two opposite reasons,
calling for opposite remedies:

  * the network has CAUGHT UP with the search -- it already predicts what the
    search would conclude, and generating more games with that budget teaches it
    nothing more. The remedy is to give the search more simulations, so the
    target moves ahead again;

  * the network is still BEHIND the search but cannot catch up. The target is
    already good and more simulations would be wasted: the problem lies in the
    optimization, the capacity or the replay window.

A single quantity tells the two apart, and nothing needs training.

THE MEASUREMENT. The policy loss decomposes exactly:

    E[-sum_a pi(a) log p(a)]  =  H(pi)  +  KL(pi || p)
    \________ loss ________/     \_ 1 _/   \____ 2 ____/

The first term is the entropy of the target: it depends only on how undecided
the search is, and the network can do nothing about it -- it is the floor of the
loss. The second is the distance between network and search, and it is the only
part training can reduce. Looking at the total loss mixes the two up, which is
why a loss that does not go down does not say, on its own, whether there is
still something to learn.

References, to read the number:
    KL / H(pi) below ~0.05   the network practically is the search: raise the
                             simulations, the target is no longer a teacher
    between ~0.05 and ~0.20  normal margin, the search is still ahead
    above ~0.20              the network is far behind: the bottleneck is
                             training, not the search budget

THERE IS ALSO THE VALUE HEAD, measured the way a probabilistic forecast is: the
positions are grouped by predicted value and the mean of each group is compared
with the mean outcome actually observed. A well-calibrated head lies on the
diagonal. If it is systematically more optimistic than the outcomes, the search
inherits that optimism at every node.

USAGE
    DAMA_WEIGHTS_DIR=cal python tools/diagnose_targets.py

    DAMA_DIAG_N=20000    how many samples to use (0 = all)
    DAMA_DEVICE=cuda     where to evaluate
"""
from __future__ import annotations
import os
import pickle
import sys

import numpy as np
import torch

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import IN_PLANES
from model import PolicyValueNet

WEIGHTS  = os.environ.get("DAMA_WEIGHTS_DIR", "weights")
CHANNELS = int(os.environ.get("DAMA_CHANNELS", "96"))
N_BLOCKS = int(os.environ.get("DAMA_BLOCKS", "6"))
USE_SE   = os.environ.get("DAMA_USE_SE", "1") != "0"
DEVICE   = os.environ.get("DAMA_DEVICE", "cpu")
N_MAX    = int(os.environ.get("DAMA_DIAG_N", "20000"))
SEED     = int(os.environ.get("DAMA_DIAG_SEED", "20260801"))

NEG = -1e9


@torch.no_grad()
def main():
    # A --help that answers: these tools take no arguments, so without this
    # line they would start computing (see env_help.py).
    from env_help import help_if_requested
    help_if_requested(__doc__, __file__)
    ckpt  = os.path.join(WEIGHTS, "champion.pt")
    state = os.path.join(WEIGHTS, "train_state.pkl")
    for p in (ckpt, state):
        if not os.path.exists(p):
            raise SystemExit(f"{p} is missing: a run that has completed at least "
                             "one cycle is needed")

    net = PolicyValueNet(in_planes=IN_PLANES, channels=CHANNELS,
                                 n_blocks=N_BLOCKS, use_se=USE_SE)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.to(DEVICE).eval()

    with open(state, "rb") as f:
        buffer = list(pickle.load(f)["buffer"])
    if not buffer:
        raise SystemExit("the buffer is empty")

    # A random but reproducible sample. The buffer is ordered by cycle: without
    # shuffling only the most recent games would be looked at.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(buffer))
    if N_MAX > 0:
        idx = idx[:N_MAX]
    samples = [buffer[i] for i in idx]

    print(f"[diagnose] champion: {ckpt}")
    print(f"[diagnose] {len(samples)} positions out of {len(buffer)} in the buffer\n")

    X = torch.from_numpy(np.stack([s[0] for s in samples]))
    P = torch.from_numpy(np.stack([s[1] for s in samples]))
    M = torch.from_numpy(np.stack([s[2] for s in samples]))
    Z = torch.from_numpy(np.array([s[3] for s in samples], np.float32))

    ce = ent = 0.0
    agree = 0
    n_legal = 0
    v_pred, v_true = [], []
    n = 0
    for i in range(0, len(X), 4096):
        xb = X[i:i+4096].to(DEVICE)
        pb = P[i:i+4096].to(DEVICE)
        mb = M[i:i+4096].to(DEVICE)
        zb = Z[i:i+4096].to(DEVICE)
        logits, value, _wdl = net(xb)
        logp = torch.log_softmax(logits.masked_fill(mb == 0, NEG), dim=1)

        ce  += float(-(pb * logp).sum())
        ent += float(-(pb * torch.log(pb.clamp_min(1e-12))).sum())
        # The move the network would play on its own against the one the search
        # visited most: the same question as the KL, but in the unit that really
        # matters at the board, the moves.
        agree += int((logp.argmax(dim=1) == pb.argmax(dim=1)).sum())
        n_legal += int((mb > 0).sum())
        v_pred.append(value.squeeze(1).cpu())
        v_true.append(zb.cpu())
        n += len(xb)

    ce, ent = ce / n, ent / n
    kl = ce - ent
    v_pred = torch.cat(v_pred).numpy()
    v_true = torch.cat(v_true).numpy()

    print("  POLICY")
    print(f"    loss             {ce:.4f} nats")
    print(f"    entropy H(pi)    {ent:.4f} nats   <- floor, not reducible")
    print(f"    KL(pi || net)    {kl:.4f} nats   <- the only learnable part")
    print(f"    KL / H           {kl / max(ent, 1e-9):.3f}")
    print(f"    mean legal moves             {n_legal / n:.2f}")
    print(f"    effective moves in target    {np.exp(ent):.2f}")
    print(f"    the network picks the most visited move in "
          f"{100.0 * agree / n:.1f}% of cases")

    r = kl / max(ent, 1e-9)
    print()
    if r < 0.05:
        print("    READING: the network has caught up with the search. With this")
        print("    budget self-play teaches it almost nothing more, and further")
        print("    cycles at the same parameters would go round in circles. The")
        print("    lever is the SIMULATIONS: more search puts the target ahead again.")
    elif r < 0.20:
        print("    READING: normal margin, the search is still ahead of the")
        print("    network. Raising the simulations would help playing strength but")
        print("    it is not what is blocking learning.")
    else:
        print("    READING: the network is far behind a target that is already")
        print("    good. More simulations would be wasted: the bottleneck is")
        print("    training (epochs, step size, replay window, network capacity).")

    print("\n  VALUE  (grouped by predicted value)")
    print("    predicted    mean outcome    positions")
    edges = np.array([-1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0001])
    weighted_err, tot = 0.0, 0
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (v_pred >= a) & (v_pred < b)
        k = int(sel.sum())
        if k == 0:
            continue
        mp, mv = float(v_pred[sel].mean()), float(v_true[sel].mean())
        print(f"    [{a:+.1f},{b:+.1f})   {mp:+.3f}  ->  {mv:+.3f}   {k:7d}")
        weighted_err += abs(mp - mv) * k
        tot += k
    print(f"    mean calibration gap         : {weighted_err / max(tot, 1):.3f}")
    print(f"    mean absolute error          : {np.abs(v_pred - v_true).mean():.3f}")
    print(f"    mean outcome in the buffer   : {v_true.mean():+.3f}"
          f"   (draws: {100.0 * (v_true == 0).mean():.1f}%)")
    print("\n    A calibrated head lies on the diagonal. A systematic gap")
    print("    upwards is optimism the search inherits at every node.")


if __name__ == "__main__":
    main()
