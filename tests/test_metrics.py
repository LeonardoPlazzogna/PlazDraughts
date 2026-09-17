"""
Tests of the metrics collection (metrics.py) and of the alpha-beta opponents
(players.AlphaBetaPlayer).

The most important check is the MONOTONICITY of the alpha-beta ladder: if a
greater depth did not beat a smaller one, the anchors would measure nothing and
the strength curve in the thesis would be meaningless.
"""
from __future__ import annotations
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from dama import Position                                   # noqa: E402
from metrics import (MetricsCsv, CycleRow, aggregate,   # noqa: E402
                     entropy_of)
from players import RandomPlayer, GreedyPlayer, AlphaBetaPlayer    # noqa: E402

ok = True


def check(cond, msg):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        ok = False


def test_entropy():
    print("--- entropy ---")
    check(abs(entropy_of([1.0])) < 1e-9, "certain distribution -> entropy 0")
    check(abs(entropy_of([0.5, 0.5]) - 0.6931) < 1e-3,
          "two equally likely outcomes -> ln(2)")
    check(entropy_of([0.25] * 4) > entropy_of([0.7, 0.1, 0.1, 0.1]),
          "uniform has more entropy than concentrated")
    check(abs(entropy_of([0.5, 0.5, 0.0])) - 0.6931 < 1e-3,
          "zeros do not produce NaN")


def test_aggregate():
    print("--- aggregation of the game statistics ---")
    games = [
        {"plies": 50, "z_white": 1, "truncated": False, "no_progress": False,
         "captures": 10, "promotions": 1, "entropy": 1.0, "value_mae": 0.5},
        {"plies": 100, "z_white": 0, "truncated": True, "no_progress": False,
         "captures": 20, "promotions": 3, "entropy": 2.0, "value_mae": 0.7},
        {"plies": 80, "z_white": -1, "truncated": False, "no_progress": True,
         "captures": 15, "promotions": 2, "entropy": 1.5, "value_mae": 0.6},
    ]
    st = aggregate(games)
    check(st.games == 3, "game count")
    check(abs(st.plies_mean - 76.667) < 0.01, f"mean length ({st.plies_mean:.2f})")
    check(st.plies_max == 100, "maximum length")
    check((st.white_wins, st.black_wins, st.draws) == (1, 1, 1), "outcomes kept apart")
    check(st.truncated == 1 and st.no_progress == 1, "termination reasons")
    check(abs(st.captures_mean - 15.0) < 1e-9, "mean captures")
    check(abs(st.entropy_mean - 1.5) < 1e-9, "mean entropy")
    check(abs(st.draw_rate - 1 / 3) < 1e-9, "draw rate")
    check(aggregate([]).games == 0, "empty list raises no exception")


def test_csv():
    print("--- CSV: creation, append, resume ---")
    with tempfile.TemporaryDirectory() as td:
        log = MetricsCsv(td)
        check(os.path.exists(log.path), "file created with the header")
        log.append(CycleRow(cycle=1, loss=2.5, arena_score=0.6, promoted=1))
        log.append(CycleRow(cycle=2, loss=2.1, arena_score=0.4, promoted=0))

        rows = list(csv.DictReader(open(log.path)))
        check(len(rows) == 2, f"two rows written ({len(rows)})")
        check(rows[0]["cycle"] == "1" and rows[1]["cycle"] == "2", "cycles in order")
        check(rows[0]["timestamp"] != "", "timestamp filled in automatically")
        check("entropy_mean" in rows[0] and "t_selfplay" in rows[0],
              "diagnostic columns present")

        # a second instance (resume) must not rewrite the header
        log2 = MetricsCsv(td)
        log2.append(CycleRow(cycle=3))
        rows2 = list(csv.DictReader(open(log2.path)))
        check(len(rows2) == 3, f"append after the resume ({len(rows2)} rows)")
        check([r["cycle"] for r in rows2] == ["1", "2", "3"],
              "history preserved across the restart")


def _duel(a, b, n=8, max_plies=160):
    tot = 0.0
    for g in range(n):
        aw = (g % 2 == 0)
        rng = np.random.default_rng(500 + g)
        pos = Position()
        plies = 0
        while not pos.is_terminal() and plies < max_plies:
            p = a if (pos.turn == 1) == aw else b
            pos = pos.play(p.move(pos, rng))
            plies += 1
        r = pos.result() or 0
        ar = r if aw else -r
        tot += 1.0 if ar > 0 else (0.5 if ar == 0 else 0.0)
    return tot / n


def test_ab_ladder():
    print("--- alpha-beta ladder: monotonicity ---")
    s_gr = _duel(GreedyPlayer(), RandomPlayer())
    check(s_gr > 0.5, f"greedy beats random ({s_gr:.2f})")

    s_ab2 = _duel(AlphaBetaPlayer(2), GreedyPlayer())
    check(s_ab2 >= 0.5, f"ab2 is not worse than greedy ({s_ab2:.2f})")

    s_ab3 = _duel(AlphaBetaPlayer(3), AlphaBetaPlayer(2))
    check(s_ab3 > 0.5, f"ab3 beats ab2 ({s_ab3:.2f})")

    # quiescence is what makes an even depth sensible: without it, ab2 came out
    # WEAKER than greedy (measured 0.40)
    check(AlphaBetaPlayer(2).MAX_EXT > 0, "quiescence active")

    # deterministic for the same seed
    p = AlphaBetaPlayer(3)
    pos = Position()
    m1 = p.move(pos, np.random.default_rng(1))
    m2 = p.move(pos, np.random.default_rng(1))
    check(m1 == m2, "same seed -> same move")


def test_tt_equivalence():
    print("--- the transposition table does not change the play ---")
    # An essential check: a table that stores values obtained under pruning
    # WITHOUT recording their bound type returns wrong numbers when read back
    # with a different alpha-beta window. The anchor would change behavior
    # silently, and the strength curve with it.
    rng = np.random.default_rng(11)
    rp = RandomPlayer()
    positions = []
    for _ in range(14):
        pos = Position()
        for _ in range(int(rng.integers(4, 40))):
            if pos.is_terminal():
                break
            pos = pos.play(rp.move(pos, rng))
        if not pos.is_terminal() and len(pos.legal_moves()) >= 2:
            positions.append(pos)

    for depth in (3, 4):
        diff = 0
        for pos in positions:
            a = AlphaBetaPlayer(depth, use_tt=False).move(pos, np.random.default_rng(1))
            b = AlphaBetaPlayer(depth, use_tt=True).move(pos, np.random.default_rng(1))
            if a != b:
                diff += 1
        check(diff == 0,
              f"depth {depth}: same move with and without the table "
              f"({len(positions) - diff}/{len(positions)})")


def main():
    test_entropy()
    test_aggregate()
    test_csv()
    test_ab_ladder()
    test_tt_equivalence()
    print("\n>>> " + ("METRICS AND ANCHORS OK" if ok else "METRICS AND ANCHORS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
