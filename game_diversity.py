"""How many of the N games of the strength test are REALLY distinct?

The question comes from a number that does not stand up: at cycle 60 of run A
the same evaluation gave 0.433 against depth-6 alpha-beta and 0.700 against
depth 8. An opponent that searches deeper cannot be easier to beat.

The suspect is the one already seen in the arena (defect 5.1 in
docs/results.md): the champion plays DETERMINISTICALLY -- search without noise
at the root, then the most visited move -- so two games with the same colors can
differ only if the OPPONENT plays differently. The only source of variety is
alpha-beta's random tie-break between moves of equal value. If ties are rare,
the sixty declared games are worth very few, and the confidence interval
computed on sixty is a fiction.

Here it is measured instead of assumed: the games are played the way the
strength test plays them, and the distinct move sequences are counted.

    python game_diversity.py --weights PATH --games 60 --depth 4 --sims 50

The simulations can be lowered from the 400 used in production: they change
HOW WELL the champion plays, not the fact that it always plays the same way,
which is the only thing that matters for this count.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dama import Position
from evaluators import NetEvaluator
from mcts import MCTS
from model import PolicyValueNet
from players import AlphaBetaPlayer


def play_one_game(champion_eval, baseline, n_sims, mcts_batch, seed, champ_white):
    """A faithful replica of the body of the strength-test loop, which also keeps
    the sequence of moves so that it can be compared with the others."""
    pos = Position()
    rng = np.random.default_rng(seed)
    moves, plies = [], 0
    while not pos.is_terminal() and plies < 300:
        champ_turn = (pos.turn == 1) == champ_white
        if champ_turn:
            root = MCTS(champion_eval, n_sims=n_sims, batch_size=mcts_batch,
                        rng=rng).run(pos, add_noise=False)
            move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        else:
            move = baseline.move(pos, rng)
        moves.append(repr(move))
        pos = pos.play(move)
        plies += 1
    r = pos.result() or 0
    outcome = r if champ_white else -r
    points = 1.0 if outcome > 0 else (0.5 if outcome == 0 else 0.0)
    return "|".join(moves), points, plies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--mcts-batch", type=int, default=8)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=60 * 104729)  # as at cycle 60
    args = ap.parse_args()

    net = PolicyValueNet(channels=args.channels, n_blocks=args.blocks)
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval()
    champion_eval = NetEvaluator(net, "cpu")
    baseline = AlphaBetaPlayer(args.depth)

    print(f"champion {args.weights}")
    print(f"{args.games} games against ab{args.depth}, {args.sims} simulations")
    print()

    t0 = time.time()
    signatures, total_points, lengths = [], 0.0, []
    by_color = {True: [], False: []}
    for g in range(args.games):
        champ_white = (g % 2 == 0)
        signature, points, plies = play_one_game(
            champion_eval, baseline, args.sims, args.mcts_batch,
            args.seed + g, champ_white)
        signatures.append(signature)
        by_color[champ_white].append(signature)
        total_points += points
        lengths.append(plies)
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.games} games  ({time.time()-t0:.0f}s)")

    distinct = len(set(signatures))
    print()
    print(f"score                : {total_points/args.games:.3f}")
    print(f"declared games       : {args.games}")
    print(f"DISTINCT games       : {distinct}")
    print(f"  champion as White  : {len(set(by_color[True]))} of {len(by_color[True])}")
    print(f"  champion as Black  : {len(set(by_color[False]))} of {len(by_color[False])}")
    print(f"mean plies           : {np.mean(lengths):.1f}")

    # The declared interval and the real one. The first divides by the nominal
    # games, the second by the distinct ones: the ratio between the two says how
    # much the measurement is deceiving itself.
    p = total_points / args.games
    se_nom = (p * (1 - p) / args.games) ** 0.5
    se_eff = (p * (1 - p) / max(1, distinct)) ** 0.5
    print()
    print(f"declared standard error (over {args.games}) : {se_nom:.3f}")
    print(f"effective standard error (over {distinct}) : {se_eff:.3f}")
    if distinct < args.games:
        print(f"-> the intervals are too narrow by a factor {se_eff/se_nom:.2f}")
    if distinct >= 0.9 * args.games:
        print("-> the diversity is enough: alpha-beta's tie-break suffices")


if __name__ == "__main__":
    main()
