"""Absolute strength of ALL the saved generations, without training anything.

The ordinary strength test runs every twenty cycles, so very few points remain
from a long run. But the champion is frozen every ten cycles for the generation
ladder, and those files stay on disk: they can be replayed at any time, after
the run, without touching training.

    python tools/anchors.py weights
    python tools/anchors.py weights --games 20 --depth 6

All generations use THE SAME seed, so the comparison between them is PAIRED:
same starting positions, same opponent, only the network changes. That is why
twenty games are enough to see differences that would otherwise need a much
larger sample -- the same principle that corrected more than one conclusion in
this project (see docs/results.md, section 5.6).

Born to answer a precise question: a generation the arena had promoted turned
out to be much weaker than the one from ten cycles earlier. With three
measurement points it could not be said whether that was an isolated case or
the rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_diversity import play_one_game
from evaluators import NetEvaluator
from model import PolicyValueNet
from players import AlphaBetaPlayer


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="weights folder (contains the gen_*.pt files)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--depth", type=int, default=6, help="opponent depth")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--mcts-batch", type=int, default=8)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--every", type=int, default=10,
                    help="cycles between one generation and the next (for the label)")
    ap.add_argument("--seed", type=int, default=6_283_740)
    ap.add_argument("--save", default="", help="JSON file to write the results to")
    args = ap.parse_args()

    # One thread per process: whoever wants to parallelize runs several processes
    # on subsets of generations, instead of making them contend for the same CPU.
    torch.set_num_threads(1)

    gens = sorted(f for f in os.listdir(args.folder)
                  if f.startswith("gen_") and f.endswith(".pt"))
    if not gens:
        raise SystemExit(f"no gen_*.pt file in {args.folder}")

    baseline = AlphaBetaPlayer(args.depth)
    print(f"{len(gens)} generations, {args.games} games each "
          f"against {baseline.name}, {args.sims} simulations")
    print(f"{'cycle':>6} {'score':>10} {'distinct':>10} {'plies':>11} {'sec':>7}")

    results = []
    for name in gens:
        cycle = int(name.split("_")[1].split(".")[0]) * args.every
        net = PolicyValueNet(channels=args.channels, n_blocks=args.blocks)
        net.load_state_dict(torch.load(os.path.join(args.folder, name),
                                       map_location="cpu"))
        net.eval()
        ev = NetEvaluator(net, "cpu")

        t0, points, signatures, plies = time.time(), 0.0, [], []
        for g in range(args.games):
            s, p, pl = play_one_game(ev, baseline, args.sims, args.mcts_batch,
                                     args.seed + g, g % 2 == 0)
            points += p
            signatures.append(s)
            plies.append(pl)
        r = {"file": name, "cycle": cycle, "score": points / args.games,
             "distinct": len(set(signatures)), "games": args.games,
             "plies": sum(plies) / len(plies), "seconds": time.time() - t0}
        results.append(r)
        print(f"{cycle:>6} {r['score']:>10.3f} "
              f"{r['distinct']:>6}/{args.games:<3} {r['plies']:>11.0f} "
              f"{r['seconds']:>7.0f}", flush=True)

    # Generation 0 is the initial network: against an opponent that really
    # searches it must score zero. If it does not, the defect is in the test bench
    # and not in the networks, and the rest of the table means nothing.
    zero = next((r for r in results if r["cycle"] == 0), None)
    if zero is not None:
        verdict = "OK" if zero["score"] <= 0.05 else "SUSPECT"
        print(f"\nnegative control (initial generation): "
              f"{zero['score']:.3f}  [{verdict}]")
        if verdict == "SUSPECT":
            print("  an untrained network should not score anything:")
            print("  before reading the numbers above, check the test bench.")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"\nwritten {args.save}")


if __name__ == "__main__":
    main()
