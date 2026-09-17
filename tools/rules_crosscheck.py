r"""
Test positions to compare the capture-priority rules with another engine.

WHY. Before using an external engine as a yardstick one must know whether it
plays the SAME game. Italian draughts has a hierarchy of capture obligations
that few implementations reproduce in full, and a divergence on a single tie
case does not show up as an error: it shows up as perfectly normal games whose
results mean nothing.

Each position here isolates ONE step of the hierarchy, pitting against each
other two chains that are identical on all the previous criteria. For each one
the program prints the chains the generator finds BEFORE the filter and the one
that survives AFTER it: without the discarded ones the comparison would prove
nothing, because it would not show that there was a choice.

HOW TO USE IT. Set up the position in the other engine (in CheckerBoard:
Position -> Setup, place the pieces, choose the side to move) and look at which
capture it proposes. If it matches on all five, the family of rules is the same
and comparisons between the two engines are valid. If it diverges on even one,
every game played through that engine must be thrown away.

    python tools/rules_crosscheck.py            positions + this engine's answer
    python tools/rules_crosscheck.py --ascii    without colors

ALREADY VERIFIED against Kingsrow Italian 1.19e by Ed Gilbert, inside 64-bit
CheckerBoard: all five positions match, including the fourth, which separates
two chains identical in the number of pieces, in the moving piece and in the
number of kings captured, and tells them apart only by WHEN the king is met.
That engine therefore applies the same hierarchy down to the last step, and
direct comparisons with it are valid.

It is the only ABSOLUTE anchor available to the project: the generation ladder
says how we change, the fixed depths say how much better we are than a search we
know, but neither says where we stand against the state of the art. Kingsrow
Italian is itself a neural network plus search, with a 1.7-million-position
opening book and endgame databases up to ten pieces.
"""
from __future__ import annotations
import argparse
import os
import sys

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import Position, WHITE, W_MAN, W_KING, B_MAN, B_KING, sq
from dama.moves import _gen_captures, _apply_priority
from play import draw_board, SQUARE_NUMBER, setup_terminal


def detail(board, m) -> str:
    """The chain together with the squares of the captured pieces, in capture order.

    Start and end alone are not enough: two different chains can end on the same
    square, and then the list would seem to repeat the same move twice. What
    tells them apart is what they capture.
    """
    chain = "x".join(str(SQUARE_NUMBER[p]) for p in m.path)
    if m.promotes:
        chain += "=D"
    prey = ", ".join(
        f"{SQUARE_NUMBER[s]} ({'king' if abs(board[s]) == 2 else 'man'})"
        for s in m.captured)
    return f"{chain:<12} captures {prey}"


def build(pieces: dict) -> Position:
    b = [0] * 64
    for (r, c), p in pieces.items():
        assert (r + c) % 2 == 1, f"light square: row {r}, column {c}"
        b[sq(r, c)] = p
    return Position(b, WHITE, 0)


# Each entry: (title, what it isolates, pieces, what must happen)
TESTS = [
    (
        "1. Number of pieces",
        "two chains of different length: the longer one must be taken",
        {(5, 4): W_MAN,
         (4, 3): B_MAN, (2, 1): B_MAN,      # two-piece branch
         (4, 5): B_MAN},                     # one-piece branch
        "the 2-piece chain, not the 1-piece one",
    ),
    (
        "2. Capture with the king",
        "two one-piece chains, one for the man and one for the king:\n"
        "     with equal numbers it must capture with the KING",
        {(5, 0): W_KING, (5, 4): W_MAN,
         (4, 1): B_MAN,                       # prey of the king
         (4, 3): B_MAN},                      # prey of the man
        "the king's chain, not the man's",
    ),
    (
        "3. Number of kings captured",
        "same king, two one-piece chains, but one captures a KING:\n"
        "     with equal numbers and the same moving piece, the one capturing more kings wins",
        {(4, 3): W_KING,
         (3, 2): B_KING,                      # prey: a king
         (3, 4): B_MAN},                      # prey: a man
        "the one that captures the black king",
    ),
    (
        "4. King met first",
        "two two-piece chains, both by the king, both with ONE king\n"
        "     inside: it is decided by which of the two meets it first",
        {(4, 3): W_KING,
         (3, 2): B_KING, (1, 2): B_MAN,      # king at the FIRST jump
         (5, 4): B_MAN, (5, 6): B_KING},     # king at the SECOND jump
        "the one that captures the king first (upwards)",
    ),
    (
        "5. A man does not capture a king",
        "the man has a king and a man within reach:\n"
        "     the black king CANNOT be captured by a man, so it is not a choice",
        {(5, 2): W_MAN,
         (4, 1): B_KING,                      # untouchable for a man
         (4, 3): B_MAN},                      # the only possible prey
        "only the capture of the black man; the king does not appear among the options",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ascii", action="store_true")
    args = ap.parse_args()
    args.ascii = setup_terminal(args.ascii)

    print(__doc__.split("HOW TO USE IT")[0].strip())
    print("\nIn every position WHITE is to move.")
    print("The dark squares carry their number; chains read"
          " start x step x end.\n")

    for title, isolates, pieces, expected in TESTS:
        pos = build(pieces)
        raw = _gen_captures(pos.board, WHITE)
        legal = _apply_priority(pos.board, raw) if raw else []

        print("=" * 70)
        print(f"  {title}")
        print(f"     {isolates}")
        print("=" * 70)
        print(draw_board(pos.board, None, args.ascii))
        pieces_txt = ", ".join(
            f"{SQUARE_NUMBER[sq(r, c)]}={'king' if abs(p) == 2 else 'man'}"
            f" {'white' if p > 0 else 'black'}"
            for (r, c), p in sorted(pieces.items()))
        print(f"\n  pieces: {pieces_txt}")

        print(f"\n  chains found by the generator ({len(raw)}):")
        for m in raw:
            mark = "   <== LEGAL" if any(
                l.frm == m.frm and l.path == m.path for l in legal) else ""
            print(f"      {detail(pos.board, m)}{mark}")

        print(f"\n  after capture priority {len(legal)} move(s) remain:")
        for m in legal:
            print(f"      {detail(pos.board, m)}")
        print(f"  expected: {expected}")

        # A test that discriminates nothing is a useless test: if the generator
        # found a single chain, the position would not test the hierarchy and
        # would pass with any engine.
        if len(raw) < 2 and "does not capture" not in title:
            print("  WARNING: a single chain, this position does not discriminate")
        print()

    print("Set them up in the other engine and compare. If they all match,")
    print("the two engines play the same game and the comparison is valid.")


if __name__ == "__main__":
    main()
