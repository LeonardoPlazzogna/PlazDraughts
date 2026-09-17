"""
Italian draughts: game state, transitions and outcome.

`Position` bundles the board, the side to move and the no-progress counter
(for the draw rule); it exposes the legal moves, the state transition and the
terminal value.

STATE  : 8x8 board (dark squares), 2 piece types x 2 colors, side to move.
ACTIONS: the legal moves; the network indexes them with the compact 256-slot
         encoding of encoder.py.
OUTCOME: zero-sum. +1 win, -1 loss, 0 draw, from the point of view of a given
         color. A side with no legal moves LOSES (no pieces left, or all of
         them blocked). Draw by the no-progress rule.
"""
from __future__ import annotations

from .board import WHITE, initial_board, render
from .moves import generate_legal_moves, apply_move, Move

# Consecutive plies without a capture or a promotion after which the game is
# drawn.
#
# THIS IS NOT THE OFFICIAL RULE, and that has been checked: see
# docs/results.md, section 9. The FID rules (Chapter I, art. 10) prescribe 40
# moves by the counted player, with at least one king per side, on request of a
# player, and above all they RESET the count at every MAN move by either side --
# whereas here the count resets only on a capture or a promotion and keeps
# running while men move.
#
# The error always goes the same way: this rule ends games EARLIER than the
# official one allows. Measured on twelve self-play games with a trained
# network: 8 ended this way, and in 2 of those (25%) the official count was
# still at 37 and at 18 out of 40. The tool is tools/draw_rule_check.py.
#
# It was not corrected because changing the criterion changes the game: it
# invalidates the comparison with every run already made, and it would have to
# change in the same breath in the C++ engine and in the browser port
# (web/engine/game.js), otherwise the three would play different games.
NO_PROGRESS_DRAW = 80


class Position:
    __slots__ = ("board", "turn", "no_progress", "_legal")

    def __init__(self, board=None, turn: int = WHITE, no_progress: int = 0):
        self.board = board if board is not None else initial_board()
        self.turn = turn
        self.no_progress = no_progress
        self._legal = None   # legal moves, computed lazily (a Position never changes)

    # --- game interface ---------------------------------------------------
    def legal_moves(self) -> list[Move]:
        # Memoized: is_terminal, result, legal_actions and legal_mask all ask
        # for the same moves, and the (expensive) generation must run ONCE.
        if self._legal is None:
            self._legal = generate_legal_moves(self.board, self.turn)
        return self._legal

    def play(self, move: Move) -> "Position":
        """New position after `move` (self is not modified)."""
        nb = apply_move(self.board, move, self.turn)
        progressed = move.is_capture or move.promotes
        npg = 0 if progressed else self.no_progress + 1
        return Position(nb, -self.turn, npg)

    # --- terminal state and outcome ---------------------------------------
    def is_terminal(self) -> bool:
        if self.no_progress >= NO_PROGRESS_DRAW:
            return True
        return len(self.legal_moves()) == 0

    def result(self) -> int | None:
        """Outcome from WHITE's point of view: +1 White wins, -1 Black wins,
        0 draw. None if the game is not over.
        The side to move with no legal moves has lost."""
        if self.no_progress >= NO_PROGRESS_DRAW:
            return 0
        if len(self.legal_moves()) == 0:
            # the side to move cannot move -> it has lost
            return -self.turn  # White (+1) to move and lost -> -1, and vice versa
        return None

    def terminal_value(self, perspective: int) -> float:
        """Terminal value from the point of view of `perspective` (+1/-1),
        consistent with the training target z (relative to the observer)."""
        r = self.result()
        if r is None:
            return 0.0
        return float(r * perspective)

    # --- utilities --------------------------------------------------------
    def key(self):
        """Hashable key (board + side to move), for transpositions."""
        return (tuple(self.board), self.turn)

    def __str__(self) -> str:
        side = "White" if self.turn == WHITE else "Black"
        return f"{render(self.board)}\nTo move: {side}  no_progress={self.no_progress}"
