r"""
Reads back a game file written by kingsrow.py, sorting the games by length.

The longest games are the most telling. Losing in fifty plies means collapsing
in the opening; losing in a hundred and twenty means holding and giving way
later, and those are two different weaknesses calling for different remedies.
The score does not tell the two apart, the length does.

    python tools/read_games.py results/games/kingsrow_run_a.txt            list
    python tools/read_games.py results/games/kingsrow_run_a.txt --show 3   the 3 longest
"""
from __future__ import annotations
import argparse
import re

HEADER = re.compile(
    r"Game (\d+)\s+--\s+we play (\w+)\s+--\s+(\w+)\s+\((\d+) plies, ([^)]+)\)")


def read_games(path: str):
    games = []
    current = None
    for line in open(path, encoding="utf-8"):
        m = HEADER.search(line)
        if m:
            current = {"n": int(m.group(1)), "color": m.group(2),
                       "outcome": m.group(3), "plies": int(m.group(4)),
                       "reason": m.group(5), "moves": []}
            games.append(current)
        elif current is not None and re.match(r"\s*\d+\.\s", line):
            current["moves"].append(line.rstrip())
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--show", type=int, default=0,
                    help="print in full the N longest games")
    args = ap.parse_args()

    games = read_games(args.file)
    if not games:
        raise SystemExit("no completed game in the file (still being played?)")

    by_length = sorted(games, key=lambda p: -p["plies"])
    print(f"{len(games)} completed games\n")
    print("  no.  color   outcome  plies  ended by")
    print("  " + "-" * 52)
    for p in by_length:
        print(f"  {p['n']:3d}  {p['color']:<7} {p['outcome']:<7} {p['plies']:6d}"
              f"  {p['reason']}")

    by_outcome = {}
    for p in games:
        by_outcome.setdefault(p["outcome"], []).append(p["plies"])
    print()
    for outcome, lengths in sorted(by_outcome.items()):
        lengths.sort()
        median = lengths[len(lengths) // 2]
        print(f"  {outcome.lower():<7} {len(lengths):3d} games, "
              f"from {lengths[0]} to {lengths[-1]} plies (median {median})")

    for p in by_length[:args.show]:
        print("\n" + "=" * 60)
        print(f"Game {p['n']} -- we play {p['color']} -- {p['outcome']} "
              f"-- {p['plies']} plies, {p['reason']}")
        print("=" * 60)
        print("\n".join(p["moves"]))


if __name__ == "__main__":
    main()
