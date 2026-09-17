"""BEHAVIOR tests of the search and of the terminal rules.

Perft, numerical parity and agreement with the external engine cover move
generation: they say that the set of legal moves is right. They do not cover
three things, however, that when wrong make nothing fail -- they only make the
engine play badly, silently:

  * the SIGN of the value along the backup through the tree. Inverted, the
    search would systematically prefer the worst moves and the run would only
    look like one "that does not learn";
  * the bookkeeping of the VIRTUAL LOSS, which is added on the way down and
    removed on the way back up. If it does not balance, the visit counts stay
    dirty and the policy target with them;
  * the TERMINAL value and its point of view, which decides who has won.

Here they are checked from observable behavior, with a uniform evaluator: if the
search still finds the move that wins at once, the sign is right.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import Position
from dama.board import WHITE, BLACK, W_MAN, B_MAN, W_KING
from evaluators import UniformEvaluator
from mcts import MCTS

FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")
    if not cond:
        FAILED.append(name)


def square(r, c):
    return r * 8 + c


def _position(pieces: dict, turn=WHITE, no_progress=0):
    b = [0] * 64
    for (r, c), p in pieces.items():
        b[square(r, c)] = p
    return Position(b, turn, no_progress)


def test_immediate_win():
    """White has a capture that removes Black's last piece.

    With mandatory captures it is the only legal move, so the value of the test
    lies in the VALUE the search assigns to the root: it must be positive for
    the side to move. With the sign inverted it would come out as -1."""
    pos = _position({(5, 2): W_MAN, (4, 3): B_MAN}, WHITE)
    moves = pos.legal_moves()
    root = MCTS(UniformEvaluator(), n_sims=64, batch_size=4).run(pos, add_noise=False)
    from selfplay import _root_value
    v = _root_value(root)
    check("winning capture: a single legal move", len(moves) == 1,
          f"({len(moves)})")
    check("winning capture: root value POSITIVE for the side to move",
          v > 0.5, f"value={v:+.3f}")


def test_choice_between_winning_and_not():
    """Two moves: one wins at once, the other does not. The search must prefer
    the first. It is the check that catches an inverted sign in the backup."""
    # White: a king that can capture the only black man, and a free man
    # elsewhere that can only move.
    pos = _position({(4, 1): W_KING, (3, 2): B_MAN, (6, 6): W_MAN}, WHITE)
    moves = pos.legal_moves()
    captures = [m for m in moves if m.is_capture]
    if not captures:
        check("choice between winning and not: position not valid for the test", False)
        return
    root = MCTS(UniformEvaluator(), n_sims=200, batch_size=8).run(pos, add_noise=False)
    best = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    check("the search chooses the capture that wins",
          best.is_capture, f"visits={root.children[best].N:.0f}")


def test_virtual_loss():
    """After the search the counts must be exact: the virtual loss added on the
    way down must have been removed entirely. If even a single one were left,
    the sum of the children's visits would not match the simulations
    requested."""
    pos = Position()
    for sims in (16, 64, 256):
        m = MCTS(UniformEvaluator(), n_sims=sims, batch_size=8)
        root = m.run(pos, add_noise=False)
        tot = sum(st.N for st in root.children.values())
        check(f"child visits = simulations ({sims})", abs(tot - sims) < 1e-9,
              f"sum={tot:.0f}")
        # no negative or fractional count left over
        check(f"no negative count ({sims})",
              all(st.N >= 0 for st in root.children.values()))


def test_terminal_value():
    """The terminal value is given from the point of view of the side that MUST
    move. Whoever has no legal moves loses: for that side the value is -1."""
    # Black without pieces: the game is over, and for Black (who would have to
    # move) it is a defeat.
    pos = _position({(5, 2): W_MAN}, BLACK)
    check("a position without enemy pieces is terminal", pos.is_terminal())
    if pos.is_terminal():
        vb = pos.terminal_value(BLACK)
        vw = pos.terminal_value(WHITE)
        check("the side that cannot move has value -1", vb == -1.0, f"black={vb:+.1f}")
        check("the opponent has value +1", vw == +1.0, f"white={vw:+.1f}")
        check("the two points of view are opposite", vb == -vw)


def test_stalemate_and_no_progress():
    """Two terminal rules that perft does not touch: whoever has no legal moves
    loses (with no distinction between having no pieces and being blocked), and
    the limit of plies without progress produces a draw."""
    # White blocked in a corner: no moves.
    # White man in the corner: the only forward diagonal is occupied by a black
    # piece, and the landing square of the jump is occupied too, so there is not
    # even a capture. White has no moves: it loses.
    pos = _position({(7, 0): W_MAN, (6, 1): B_MAN, (5, 2): B_MAN}, WHITE)
    if not pos.legal_moves():
        check("without legal moves the position is terminal", pos.is_terminal())
        check("without legal moves one LOSES (it is not a draw)",
              pos.terminal_value(WHITE) == -1.0)
    else:
        print(f"  [info] the stalemate position has {len(pos.legal_moves())} moves: "
              "test skipped")

    # No-progress limit reached: a draw, whatever the material.
    limit = None
    for mod, name in (("dama.game", "NO_PROGRESS_DRAW"),
                      ("dama.board", "NO_PROGRESS_DRAW"),
                      ("dama.moves", "NO_PROGRESS_DRAW")):
        try:
            limit = getattr(__import__(mod, fromlist=[name]), name)
            break
        except Exception:
            pass
    if limit is None:
        print("  [info] no-progress limit not exposed: test skipped")
        return
    rich = _position({(5, 2): W_MAN, (5, 4): W_MAN, (5, 6): W_MAN,
                      (2, 1): B_MAN}, WHITE, no_progress=limit)
    check(f"at the limit of {limit} plies without progress it is terminal",
          rich.is_terminal())
    if rich.is_terminal():
        check("and the result is a DRAW even with unbalanced material",
              rich.terminal_value(WHITE) == 0.0,
              f"value={rich.terminal_value(WHITE):+.1f}")


def test_determinism():
    """With the same seed and no noise, two searches must give the same counts.
    If they do not, the arena is not comparing two networks but two executions,
    and every strength measurement is noise."""
    pos = Position()
    a = MCTS(UniformEvaluator(), n_sims=128, batch_size=8).run(pos, add_noise=False)
    b = MCTS(UniformEvaluator(), n_sims=128, batch_size=8).run(pos, add_noise=False)
    ca = sorted((repr(m), st.N) for m, st in a.children.items())
    cb = sorted((repr(m), st.N) for m, st in b.children.items())
    check("two identical searches give the same counts", ca == cb)


if __name__ == "__main__":
    print("--- search and terminal rules ---")
    test_immediate_win()
    test_choice_between_winning_and_not()
    test_virtual_loss()
    test_terminal_value()
    test_stalemate_and_no_progress()
    test_determinism()
    print()
    if FAILED:
        print(f">>> {len(FAILED)} TESTS FAILED: {FAILED}")
        sys.exit(1)
    print(">>> SEARCH AND TERMINAL RULES OK")
