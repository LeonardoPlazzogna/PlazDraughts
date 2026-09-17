"""
Tests of the generation ladder (ladder.py).

Checks the Elo math on cases with a known value, the behavior on extreme scores
(which would give infinity), persistence on disk and the combination of several
anchors.
"""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ladder import Ladder, elo_diff, expected_score, SCORE_CAP   # noqa: E402

ok = True


def check(cond, msg):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        ok = False


def test_elo_math():
    print("--- Elo math ---")
    check(abs(elo_diff(0.5)) < 1e-9, "score 0.5 -> 0 Elo")
    # classic values of the Elo formula
    check(abs(elo_diff(0.75) - 190.85) < 0.5,
          f"score 0.75 -> ~+191 Elo (got {elo_diff(0.75):.2f})")
    check(abs(elo_diff(0.25) + 190.85) < 0.5,
          f"score 0.25 -> ~-191 Elo (got {elo_diff(0.25):.2f})")
    check(elo_diff(0.9) > elo_diff(0.8) > elo_diff(0.6) > 0,
          "monotonically increasing in the score")
    check(abs(elo_diff(0.6) + elo_diff(0.4)) < 1e-9,
          "antisymmetric: elo(s) = -elo(1-s)")

    # the extremes must not give infinity or NaN
    for s in (0.0, 1.0):
        v = elo_diff(s)
        check(abs(v) < 1e4 and v == v, f"score {s} bounded ({v:+.0f}), not infinite")
    check(elo_diff(1.0) == elo_diff(SCORE_CAP),
          "score 1.0 clipped exactly to SCORE_CAP")

    # consistent inverse
    for d in (-300.0, -50.0, 0.0, 120.0, 400.0):
        check(abs(elo_diff(expected_score(d)) - d) < 1e-6,
              f"elo_diff(expected_score({d:+.0f})) = {d:+.0f}")


def test_persistence():
    print("--- persistence and resume ---")
    with tempfile.TemporaryDirectory() as td:
        written = []
        lad = Ladder(td)
        check(lad.is_empty(), "a new ladder is empty")
        check(lad.latest_elo() == 0.0, "initial Elo 0 on an empty ladder")

        e0 = lad.add(0, 0.0, lambda p: written.append(p) or open(p, "wb").close())
        e1 = lad.add(10, 150.0, lambda p: written.append(p) or open(p, "wb").close())
        check(e0["gen"] == 0 and e1["gen"] == 1, "numbering of the generations")
        check(e1["file"] == "gen_001.pt", f"expected file name ({e1['file']})")
        check(all(os.path.exists(p) for p in written), "the snapshots are written")
        check(abs(lad.latest_elo() - 150.0) < 1e-9, "latest_elo = last recorded")

        # reload from disk
        lad2 = Ladder(td)
        check(len(lad2.entries) == 2, "ladder read back from disk")
        check(abs(lad2.latest_elo() - 150.0) < 1e-9, "Elo preserved across the restart")
        check(not os.path.exists(os.path.join(td, "ladder.json.tmp")),
              "no temporary file left over (atomic write)")

        # anchors: the most recent first
        anc = lad2.anchors(2)
        check([a["gen"] for a in anc] == [1, 0], "anchors ordered from the most recent")
        check(len(lad2.anchors(10)) == 2, "asking for more anchors than exist does not break")

        # corrupted file -> clean restart, not an exception
        with open(os.path.join(td, "ladder.json"), "w") as f:
            f.write("{ not valid json")
        lad3 = Ladder(td)
        check(lad3.is_empty(), "corrupted ladder -> empty restart without exception")


def test_estimate():
    print("--- Elo estimate from several anchors ---")
    with tempfile.TemporaryDirectory() as td:
        lad = Ladder(td)
        a0 = {"gen": 0, "cycle": 0, "file": "x", "elo": 0.0}
        a1 = {"gen": 1, "cycle": 10, "file": "y", "elo": 200.0}

        # beating a 200 anchor at 75% -> ~391
        elo, det = lad.estimate_elo([(a1, 0.75)])
        check(abs(elo - (200.0 + 190.85)) < 1.0, f"a single anchor ({elo:.0f})")
        check(len(det) == 1 and "gen1" in det[0], "text detail produced")

        # two agreeing anchors -> mean
        elo2, _ = lad.estimate_elo([(a0, 0.9), (a1, 0.5)])
        expected = ((0.0 + elo_diff(0.9)) + (200.0 + 0.0)) / 2
        check(abs(elo2 - expected) < 1e-6, f"mean of two anchors ({elo2:.0f})")

        # saturated score flagged
        _elo3, det3 = lad.estimate_elo([(a0, 1.0)])
        check("saturated" in det3[0], "extreme score flagged as saturated")

        check(lad.estimate_elo([])[0] == 0.0, "no anchor -> 0, without division by zero")


def main():
    test_elo_math()
    test_persistence()
    test_estimate()
    print("\n>>> " + ("LADDER OK" if ok else "LADDER FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
