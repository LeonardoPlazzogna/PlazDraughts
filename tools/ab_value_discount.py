r"""
A/B test of the value discount, on identical games labelled in two ways.

THE DESIGN. The discount changes only the LABELS, not the games: the same
position, in the same game, gets z or z*gamma^d depending on how far the end is.
So self-play is generated ONCE and labelled twice. The two candidates start from
the same champion, see the same positions in the same order, and differ by a
single number.

It is the same principle as tools/sweep_train.py, and it avoids the mistake of
running two separate runs: those would also differ in the games generated, and
with a few dozen games the variance between two sets is easily larger than the
effect being looked for.

WHAT IS MEASURED, and why Elo is not enough. The hypothesis is not "the network
gets stronger" but "the network stops going round in circles once it has won". A
difference like that shows up in the way it plays long before it shows in a
tournament: so the primary judgement is the share of king moves wasted going
back and forth, measured in games against fixed-depth alpha-beta (see the second
attempt below), together with the score and the game length. The direct
comparison between the two candidates is there too (tools/compare.py), but with
few games it will almost certainly say "indistinguishable", and that is
information, not a failure.

A LIMIT, stated: drawn games are labelled zero, and a discounted zero is still
zero. If a won position is squandered all the way to a no-progress draw, this
mechanism does not recover it. It acts inside the games someone wins, making it
more attractive to win them sooner. If the measured effect were nil, this is the
first thing to look at.

USAGE
    python tools/ab_value_discount.py --weights FOLDER --games 120 --gamma 0.99

TWO WAYS TO GET A FALSE NEGATIVE, both worth avoiding before running this:

  * A HEALTHY PATIENT. A champion that does not show the defect -- no king
    moves wasted going back and forth -- has nothing for the discount to
    correct, and the comparison can only say "no effect".
  * A BLIND BENCH. Measuring on already won endgames played by the network
    against itself does not see the defect either: there the network starts
    from a position already won and converts it. The defect lives in long,
    balanced endgames against an opponent that does not cooperate, which is why
    the bench here plays against fixed-depth alpha-beta.

MIND THE DENOMINATOR. Pooling all the moves instead of averaging per game can
reverse the verdict: the discounted candidate plays shorter games, so it
contributes fewer moves, and the missing ones come precisely from the long
phases where the wandering is worst. A 250-ply game would weigh as much as three
of 80.

LIMIT: against an opponent the network already beats almost every time the score
has no room to improve, so a behavior that gets better can show no gain at all.
Whether it translates into strength where the gap is real must be measured
separately.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import IN_PLANES, Position, WHITE
from model import PolicyValueNet
from evaluators import NetEvaluator
from mcts import MCTS
from selfplay import play_game
from train import train_on_samples


def load_network(ckpt: str):
    n = PolicyValueNet(in_planes=IN_PLANES, channels=96, n_blocks=6,
                               use_se=True)
    n.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return n


def generate_games(ev, n_games: int, n_sims: int, seed: int):
    """Self-play games, kept SEPARATE: without the boundaries between games the
    distance from the end cannot be rebuilt and the discount cannot be applied."""
    games = []
    t0 = time.time()
    for i in range(n_games):
        samples, z, plies, _ = play_game(ev, n_sims=n_sims, seed=seed + i)
        games.append(samples)
        if (i + 1) % 20 == 0:
            done = time.time() - t0
            print(f"   {i+1}/{n_games} games  ({done:.0f}s, "
                  f"estimated {done/(i+1)*n_games:.0f}s in total)", flush=True)
    return games


def label(games, gamma: float):
    """Applies the discount again to the same positions. The games come with
    gamma=1, so here it is enough to multiply by gamma^(distance from the end)."""
    out = []
    for samples in games:
        n = len(samples)
        for i, (X, pi, mask, z) in enumerate(samples):
            out.append((X, pi, mask, float(z * gamma ** (n - 1 - i))))
    return out


def match_vs_alphabeta(ev, n_games: int, depth: int, n_sims: int,
                       max_plies: int = 300, seed: int = 100):
    """The network against fixed-depth alpha-beta: score AND wandering.

    It is the bench that sees the defect, and the choice is not obvious -- a
    bench on already won endgames played by the network against itself does NOT
    see it, and an experiment on a blind instrument can only say "no effect".

    The difference is that there the network starts from positions already won
    and converts them; the defect lives instead in long, balanced endgames
    against an opponent that does not cooperate, which is what fixed-depth
    alpha-beta provides.
    """
    from players import AlphaBetaPlayer
    opp = AlphaBetaPlayer(depth=depth)
    back = kings = 0
    won = drawn = lost = 0
    lengths = []
    for g in range(n_games):
        rng = np.random.default_rng(seed + g)
        we_white = (g % 2 == 0)
        pos, plies, prev = Position(), 0, None
        while not pos.is_terminal() and plies < max_plies:
            if (pos.turn == WHITE) == we_white:
                root = MCTS(ev, n_sims=n_sims, c_puct=1.5, batch_size=8,
                            rng=rng).run(pos, add_noise=False)
                children = list(root.children.items())
                if plies < 10:
                    w = np.array([st.N for _, st in children], float)
                    mv = children[int(rng.choice(len(children), p=w / w.sum()))][0]
                else:
                    mv = max(children, key=lambda kv: kv[1].N)[0]
                if mv.by_king and not mv.is_capture:
                    kings += 1
                    if prev is not None and prev.to == mv.frm and prev.frm == mv.to:
                        back += 1
                prev = mv
            else:
                mv = opp.move(pos, rng)
            pos = pos.play(mv)
            plies += 1
        r = pos.result() or 0
        ours = r if we_white else -r
        won += ours > 0; drawn += ours == 0; lost += ours < 0
        lengths.append(plies)
    return {"w": won, "d": drawn, "l": lost, "score": (won + 0.5 * drawn) / max(1, n_games),
            "back": back, "kings": kings,
            "share": 100 * back / max(1, kings),
            "length": sum(lengths) / max(1, len(lengths))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("DAMA_WEIGHTS_DIR", "weights"))
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--sims-test", type=int, default=200)
    ap.add_argument("--ab", type=int, default=4,
                    help="depth of the alpha-beta used as the test bench")
    ap.add_argument("--ab-games", type=int, default=14,
                    help="games against alpha-beta for each candidate")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", default="ab_value_discount")
    args = ap.parse_args()

    ckpt = os.path.join(args.weights, "champion.pt")
    base = load_network(ckpt)
    base.eval()
    print(f"[ab] starting champion: {ckpt}")
    # The games depend on the champion and the seeds, not on the discount: they
    # are generated once and reused. It lets the experiment be run again with
    # different gammas without paying half an hour of self-play each time -- and
    # guarantees that comparisons made at different times see the SAME data.
    import pickle
    cache = os.path.join(args.out, f"games_{args.games}_{args.sims}_{args.seed}.pkl")
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(cache):
        games = pickle.load(open(cache, "rb"))
        print(f"[ab] reusing the games already generated in {cache}")
    else:
        print(f"[ab] generating {args.games} games at {args.sims} simulations "
              f"(once only, both candidates use them)")
        games = generate_games(NetEvaluator(base, "cpu"), args.games, args.sims, args.seed)
        pickle.dump(games, open(cache, "wb"))
    n_pos = sum(len(p) for p in games)
    print(f"[ab] {n_pos} positions from {len(games)} games\n")

    results = {}
    for name, gamma in (("no discount", 1.0), (f"gamma {args.gamma}", args.gamma)):
        samples = label(games, gamma)
        z = np.array([c[3] for c in samples])
        net = load_network(ckpt)
        torch.manual_seed(args.seed)
        train_on_samples(net, samples, epochs=args.epochs, batch_size=256,
                         lr=1e-3, device="cpu")
        net.eval()
        path = os.path.join(args.out, f"{'no_discount' if gamma == 1.0 else 'discount'}.pt")
        torch.save(net.state_dict(), path)
        print(f"\n  === {name} ===")
        print(f"     labels: mean |z| {np.abs(z).mean():.3f}, "
              f"zero {100*(z == 0).mean():.0f}%")
        m = match_vs_alphabeta(NetEvaluator(net, "cpu"), args.ab_games,
                               args.ab, args.sims_test)
        print(f"     against ab{args.ab}: {m['w']}W {m['d']}D {m['l']}L  "
              f"-> score {m['score']:.3f}, {m['length']:.0f} mean plies")
        print(f"     king moves {m['kings']}, back and forth {m['back']} "
              f"({m['share']:.1f}%)")
        results[name] = (m, path)

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print(f"   {'':<16} {'score':>10} {'back-forth':>11}")
    for name, (m, _) in results.items():
        print(f"   {name:<16} {m['score']:10.3f} {m['share']:10.1f}%")
    a, b = list(results.values())
    d_s = b[0]["score"] - a[0]["score"]
    d_b = b[0]["share"] - a[0]["share"]
    print(f"\n   difference (discount minus none): score {d_s:+.3f}, "
          f"back-and-forth {d_b:+.1f} percentage points")
    print("\n  For the direct comparison between the two candidates:")
    print(f"    python tools/compare.py {a[1]} {b[1]}")


if __name__ == "__main__":
    main()
