r"""
How much of the policy target no network could ever predict.

THE QUESTION. The divergence between network and search has stayed at about
0.13 nats for two hundred cycles, and it does not go down even when training
fifteen times longer. Before concluding that the network lacks capacity,
something simpler has to be ruled out: that part of that residual is
IRREDUCIBLE.

The target is not a function of the position. It is the visit distribution of a
search in which, at the root, the priors are mixed with Dirichlet noise drawn at
random every time. Two searches on the SAME position therefore give different
targets, and no network can predict that difference: it depends on a draw, not
on the board.

HOW IT IS MEASURED. For each position the search is run K times with different
draws, giving pi_1..pi_K. The best possible prediction is their mean pi_bar --
it is the point that minimizes the expected divergence -- so the floor is

    E[ KL(pi_k || pi_bar) ]

If this number is close to 0.13, the network has already extracted all the
signal and the residual is the draw: more capacity or more epochs cannot help at
all. If instead it is much smaller, the residual is signal not yet learned, and
capacity is back in play.

THE CONTROL. Everything is repeated WITHOUT noise. This search is then
deterministic -- same position, same network, same budget, same visits -- so the
control must come out at exactly 0.0000 (measured: 0.0000 on 150 positions). Any
other value would mean that the variability comes from something else, and the
main measurement would not mean what it is believed to mean.

USAGE
    DAMA_WEIGHTS_DIR=diag DAMA_DEVICE=cuda python tools/target_noise.py
    DAMA_TN_POS=150     positions to test
    DAMA_TN_K=6         searches per position
    DAMA_TN_SIMS=400    simulations per search
    DAMA_TN_SEED=20260808
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import IN_PLANES, Position
from model import PolicyValueNet
from evaluators import NetEvaluator
from mcts import MCTS
from selfplay import visit_policy_target, select_move

N_POS = int(os.environ.get("DAMA_TN_POS", "150"))
K = int(os.environ.get("DAMA_TN_K", "6"))
SIMS = int(os.environ.get("DAMA_TN_SIMS", "400"))
DEVICE = os.environ.get("DAMA_DEVICE", "cpu")
WEIGHTS = os.environ.get("DAMA_WEIGHTS_DIR", "weights")
SEED = int(os.environ.get("DAMA_TN_SEED", "20260808"))


def positions(ev, n: int, rng):
    """Positions collected by playing, so they come from the same distribution
    the network is trained on -- not from an artificial sampling."""
    out, pos, plies = [], Position(), 0
    while len(out) < n:
        if pos.is_terminal() or plies > 250:
            pos, plies = Position(), 0
            continue
        out.append(pos)
        root = MCTS(ev, n_sims=100, c_puct=1.5, batch_size=8, rng=rng).run(
            pos, add_noise=True)
        pos = pos.play(select_move(root, 1.0 if plies < 12 else 0.0, rng))
        plies += 1
    return out


def mean_kl(ev, pos_list, with_noise: bool):
    """Mean divergence between one target and the mean of the targets on the
    same position: the floor that no function of the board can beat."""
    tot, n = 0.0, 0
    for i, pos in enumerate(pos_list):
        pis = []
        for k in range(K):
            rng = np.random.default_rng(SEED + 1000 * i + k)
            root = MCTS(ev, n_sims=SIMS, c_puct=1.5, batch_size=8,
                        rng=rng).run(pos, add_noise=with_noise)
            pis.append(visit_policy_target(pos, root))
        P = np.stack(pis)                      # [K, actions]
        mean = P.mean(axis=0)
        m = mean > 0
        for k in range(K):
            p = P[k]
            sel = m & (p > 0)
            tot += float((p[sel] * np.log(p[sel] / mean[sel])).sum())
            n += 1
        if (i + 1) % 25 == 0:
            print(f"   {i+1}/{len(pos_list)} positions  "
                  f"({'with' if with_noise else 'without'} noise): "
                  f"{tot/n:.4f} nats", flush=True)
    return tot / max(1, n)


def main():
    # A --help that answers: these tools take no arguments, so without this
    # line they would start computing (see env_help.py).
    from env_help import help_if_requested
    help_if_requested(__doc__, __file__)
    ckpt = os.path.join(WEIGHTS, "champion.pt")
    net = PolicyValueNet(in_planes=IN_PLANES, channels=96, n_blocks=6,
                                 use_se=True)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.eval()
    ev = NetEvaluator(net, DEVICE)
    print(f"[noise] champion {ckpt}, {N_POS} positions x {K} searches "
          f"of {SIMS} simulations\n")

    rng = np.random.default_rng(SEED)
    pos_list = positions(ev, N_POS, rng)

    print("  main measurement: targets WITH Dirichlet noise")
    with_noise = mean_kl(ev, pos_list, True)
    print("\n  control: same positions WITHOUT noise")
    without_noise = mean_kl(ev, pos_list, False)

    print("\n" + "=" * 58)
    print(f"  irreducible floor (with noise)       : {with_noise:.4f} nats")
    print(f"  residual variability (without noise) : {without_noise:.4f} nats")
    print(f"  network-search KL measured on the run: 0.1262 nats")
    print()
    share = 100 * with_noise / 0.1262
    print(f"  the draw explains about {share:.0f}% of the residual")
    if share > 70:
        print("  READING: the residual is almost all draw. The network has already")
        print("  extracted the available signal, and neither more capacity nor more")
        print("  epochs can reduce it. To lower it, act on the NOISE")
        print("  (dirichlet_eps) or on the number of simulations, not on the network.")
    elif share > 30:
        print("  READING: a substantial part is draw, but not all of it: there is")
        print("  still room for learning, narrower than it looks.")
    else:
        print("  READING: the draw explains little. The residual is signal not yet")
        print("  learned, and the capacity of the network is the main suspect again.")


if __name__ == "__main__":
    main()
