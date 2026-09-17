r"""
Material balance in QUIET positions, in the games against the external engine.

It replays the games recorded by kingsrow.py and, for each one, counts the
material from our champion's point of view in every quiet position: one where
the side to move has no capture available. With mandatory captures the count
oscillates inside every exchange, so measuring it at every ply would only say
that an exchange is under way.

These counts give the material table of chapter 8 of the thesis and the data of
the figure on the peak advantage (results/series/*_quiet_peak_*.csv). The
original script had been lost; this reconstruction reproduces all eight figures
of the table.

A KING COUNTS AS ONE PIECE. The table counts pieces, not weighted material: with
the king at 3 (the weight selfplay.py uses to adjudicate truncated games) the
figures do not match -- "never ahead" for run A goes from 15/27 to 4/27.

    python tools/quiet_material.py                   table for run A and run B
    python tools/quiet_material.py --csv             also rewrites the figure data
    python tools/quiet_material.py --king-weight 3   to see that the figures do not match
"""
from __future__ import annotations
import argparse
import csv
import io
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dama import Position                                   # noqa: E402
from dama.board import W_MAN, W_KING, B_MAN, B_KING          # noqa: E402
from dama.moves import has_capture                           # noqa: E402
from play import SQUARE_NUMBER                               # noqa: E402

HEADER = re.compile(r"Game (\d+)\s+--\s+we play (\w+)\s+--\s+(\w+)\s+\((\d+) plies")
MOVE = re.compile(r"\d+(?:[-x]\d+)*(?:=D)?")
DURABLE = 10   # consecutive quiet positions for an advantage to count as durable

# run -> (recorded games, figures published in the table of chapter 8)
RUN = {
    "A": (os.path.join("results", "games", "kingsrow_run_a.txt"),
          "peak 0 | never ahead 15/27 | durable 0/50 | final 0 / -2"),
    "B": (os.path.join("results", "games", "kingsrow_run_b.txt"),
          "peak +1 | never ahead 10/30 | durable 0/50 | final 0 / -3"),
}


def notation(m) -> str:
    """The move as kingsrow.py writes it: dark squares numbered 1..32."""
    s = "x".join(str(SQUARE_NUMBER[p]) for p in m.path) if m.is_capture \
        else f"{SQUARE_NUMBER[m.frm]}-{SQUARE_NUMBER[m.to]}"
    return s + ("=D" if m.promotes else "")


def read_games(path: str) -> list[dict]:
    out, cur = [], None
    for line in io.open(path, encoding="utf-8", errors="replace"):
        h = HEADER.search(line)
        if h:
            cur = {"n": int(h.group(1)), "us": h.group(2), "outcome": h.group(3), "moves": []}
            out.append(cur)
        elif cur is not None and re.match(r"\s*\d+\.", line):
            cur["moves"].extend(MOVE.findall(re.sub(r"^\s*\d+\.", "", line)))
    return out


def analyse(p: dict, king_weight: int) -> dict | None:
    """Replays a game; None if a move matches no legal move."""
    mat = {W_MAN: 1, W_KING: king_weight, B_MAN: -1, B_KING: -king_weight}
    sign = 1 if p["us"] == "White" else -1
    pos, quiet = Position(), []
    for tok in p["moves"]:
        choice = next((m for m in pos.legal_moves() if notation(m) == tok), None)
        if choice is None:
            return None
        pos = pos.play(choice)
        if not has_capture(pos.board, pos.turn):
            quiet.append(sign * sum(mat.get(x, 0) for x in pos.board))
    if not quiet:
        return None
    streak = best = 0               # longest run of quiet positions with balance >= 2
    for v in quiet:
        streak = streak + 1 if v >= 2 else 0
        best = max(best, streak)
    return {"outcome": p["outcome"], "peak": max(quiet), "final": quiet[-1],
            "durable": best >= DURABLE}


def summary(run: str, res: list[dict], read: int, king_weight: int) -> None:
    draws = [r for r in res if r["outcome"] == "DRAW"]
    others = [r for r in res if r["outcome"] != "DRAW"]
    print(f"\n### run {run}  (king = {king_weight})  games replayed {len(res)}/{read}"
          f" -- {len(draws)} draws, {len(others)} not drawn")
    print(f"  quiet peak, median           : {st.median(r['peak'] for r in res):+.0f}")
    print(f"  never ahead (among draws)    : {sum(1 for r in draws if r['peak'] <= 0)}"
          f" of {len(draws)}")
    print(f"  durable advantage >= 2       : {sum(1 for r in res if r['durable'])} of {len(res)}")
    print(f"  final material, median       : draws {st.median(r['final'] for r in draws):+.0f}"
          f" / others {st.median(r['final'] for r in others):+.0f}")
    print(f"  published figures            : {RUN[run][1]}")


def write_csv(name: str, rows: list[dict]) -> None:
    """Games sorted by peak, as a percentage of the sample: the figure in chapter
    8 draws this curve."""
    rows = sorted(rows, key=lambda r: (r["peak"], r["final"]))
    n = len(rows)
    with io.open(os.path.join(ROOT, "results", "series", name), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_pct", "peak", "final", "result", "durable"])
        for i, r in enumerate(rows):
            w.writerow([round(100 * i / (n - 1), 2), r["peak"], r["final"],
                        r["outcome"], "yes" if r["durable"] else "no"])
    print(f"  written results/series/{name} ({n} games)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--king-weight", type=int, default=1)
    ap.add_argument("--csv", action="store_true",
                    help="rewrite the figure data in results/series/")
    args = ap.parse_args()
    if args.csv and args.king_weight != 1:
        sys.exit("--csv only with --king-weight 1: the figure counts pieces, not weighted material.")

    analysis = {}
    for run, (games_file, _) in RUN.items():
        games = read_games(os.path.join(ROOT, games_file))
        analysis[run] = [r for r in (analyse(p, args.king_weight) for p in games) if r]
        summary(run, analysis[run], len(games), args.king_weight)

    if args.csv:
        print()
        for run, res in analysis.items():
            write_csv(f"run{run}_quiet_peak_kr.csv", res)
        # depth 8 has already been measured game by game: same format
        r8 = list(csv.DictReader(io.open(os.path.join(ROOT, "results", "series", "champion_vs_ab8.csv"),
                                         encoding="utf-8")))
        write_csv("runA_quiet_peak_ab8.csv",
                  [{"peak": int(x["max_quiet"]), "final": int(x["final_quiet"]),
                    "outcome": x["result"], "durable": x["durable_ge2"].strip().lower() == "yes"}
                   for x in r8])


if __name__ == "__main__":
    main()
