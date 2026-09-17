"""
Tests of the Italian draughts rules. Runnable: `python tests/test_rules.py`.
No dependencies (no pytest): prints PASS/FAIL and exits 1 if anything fails.
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import (
    WHITE, BLACK, W_MAN, W_KING, B_MAN, B_KING, EMPTY,
    Position, apply_move,
    encode, legal_actions, legal_mask, index_to_move, IN_PLANES,
)
from dama.board import initial_board

FAIL = 0


def check(name, cond):
    global FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAIL += 1


def eb():
    return [EMPTY] * 64


# 1) Initial position
def test_initial():
    p = Position()
    w = sum(1 for x in p.board if x > 0)
    b = sum(1 for x in p.board if x < 0)
    moves = p.legal_moves()
    check("initial: 12 white and 12 black", w == 12 and b == 12)
    check("initial: 7 legal moves for White", len(moves) == 7)
    check("initial: no capture", all(not m.is_capture for m in moves))


# 2) Mandatory capture
def test_mandatory():
    b = eb()
    b[42] = W_MAN      # (5,2)
    b[33] = B_MAN      # (4,1)  -> capturable towards (3,0)=24
    p = Position(b, WHITE)
    moves = p.legal_moves()
    check("mandatory: if there is a capture, every move is a capture",
          len(moves) > 0 and all(m.is_capture for m in moves))


# 3) Greater capture: maximum number of pieces
def test_max_capture():
    b = eb()
    b[42] = W_MAN                 # double: 42->24->10, captures 33 and 17
    b[33] = B_MAN
    b[17] = B_MAN
    b[46] = W_MAN                 # single: 46->28, captures 37
    b[37] = B_MAN
    p = Position(b, WHITE)
    moves = p.legal_moves()
    ok = (len(moves) == 1 and moves[0].frm == 42 and moves[0].to == 10
          and len(moves[0].captured) == 2)
    check("greater capture: the double is mandatory, not the single", ok)


# 4) A man does NOT capture a king (but captures a man)
def test_man_cannot_take_king():
    b = eb()
    b[42] = W_MAN
    b[33] = B_KING     # adjacent enemy king, landing (3,0)=24 free
    p = Position(b, WHITE)
    moves = p.legal_moves()
    check("a man does not capture a king",
          all(not m.is_capture for m in moves))
    # same position but with an enemy man -> capture possible
    b[33] = B_MAN
    p2 = Position(b, WHITE)
    check("a man captures a man",
          any(m.is_capture and 33 in m.captured for m in p2.legal_moves()))


# 5) Promotion with capture: promotes and stops
def test_promotion():
    b = eb()
    b[19] = W_MAN      # (2,3)
    b[10] = B_MAN      # (1,2) -> landing (0,1)=1 = last rank
    p = Position(b, WHITE)
    moves = p.legal_moves()
    ok = (len(moves) == 1 and moves[0].promotes and moves[0].to == 1
          and len(moves[0].captured) == 1)
    check("promotion: a capture that reaches the last rank promotes", ok)
    nb = apply_move(b, moves[0], WHITE)
    check("promotion: the piece on the landing square is a white king", nb[1] == W_KING)


# 6) The king captures in every direction but is not a flying king
def test_king_moves():
    b = eb()
    b[26] = W_KING     # (3,2)
    b[35] = B_MAN      # (4,3) = "backwards" for White; landing (5,4)=44
    p = Position(b, WHITE)
    check("the king captures backwards",
          any(m.is_capture and 35 in m.captured for m in p.legal_moves()))
    # not flying: piece 2 squares away, intermediate square empty -> no capture
    b2 = eb()
    b2[17] = W_KING    # (2,1)
    b2[35] = B_MAN     # (4,3) two steps away; (3,2)=26 empty
    p2 = Position(b2, WHITE)
    check("non-flying king: does not jump pieces 2 squares away",
          all(not m.is_capture for m in p2.legal_moves()))


# 7) Terminal: whoever has no moves loses
def test_terminal():
    b = eb()
    b[33] = B_MAN      # no white piece, White to move
    p = Position(b, WHITE)
    check("terminal: without moves it is terminal", p.is_terminal())
    check("terminal: White (to move, blocked) loses -> result -1",
          p.result() == -1)
    check("terminal: value from Black's point of view = +1",
          p.terminal_value(BLACK) == 1.0)


# 8) Random playout: always ends with an outcome in {-1,0,1}
def test_random_playout():
    rng = random.Random(0)
    ok_all = True
    for g in range(30):
        p = Position()
        plies = 0
        while not p.is_terminal() and plies < 1000:
            mv = rng.choice(p.legal_moves())
            p = p.play(mv)
            plies += 1
        if p.result() not in (-1, 0, 1) or plies >= 1000:
            ok_all = False
            break
    check("random playout: 30 games end with a valid outcome", ok_all)


# 9) Encoder + masking + action round-trip
def test_encoder():
    p = Position()
    x = encode(p)
    check("encoder: shape [IN_PLANES,8,8]", x.shape == (IN_PLANES, 8, 8))
    check("encoder: 12 of our men, 12 of theirs (White to move)",
          x[0].sum() == 12 and x[2].sum() == 12 and x[1].sum() == 0)

    # Black to move: canonicalization (rotate 180 + swap colors)
    pb = Position(initial_board(), BLACK)
    xb = encode(pb)
    # a black man on absolute square 1 = (0,1) must end up on 63-1=62 = (7,6)
    check("encoder: Black canonicalization (black man 1 -> canonical (7,6))",
          xb[0, 7, 6] == 1.0 and xb[0].sum() == 12 and xb[2].sum() == 12)

    # round-trip: action index -> move
    moves, idx = legal_actions(p)
    m0 = index_to_move(p, idx[0])
    check("action: index->move round-trip", m0 == moves[0])
    check("mask: sum = number of distinct legal moves", legal_mask(p).sum() == len(set(idx)))


def main():
    for t in (test_initial, test_mandatory, test_max_capture,
              test_man_cannot_take_king, test_promotion, test_king_moves,
              test_terminal, test_random_playout, test_encoder):
        print(f"--- {t.__name__} ---")
        t()
    print()
    if FAIL == 0:
        print(">>> ALL TESTS PASSED")
        sys.exit(0)
    print(f">>> {FAIL} TESTS FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
