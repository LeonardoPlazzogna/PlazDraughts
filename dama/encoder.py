"""
Italian draughts: position -> tensor encoding and the compact action space.

Network input: a [IN_PLANES=7, 8, 8] tensor, canonicalized to the point of view
of the side to move (for Black the board is rotated 180 degrees and the colors
are swapped: square s becomes 63 - s). Planes: 4 for the pieces (our and their
men and kings) plus 3 engineered ones: dark-square mask, capture available
(captures are mandatory), no-progress counter.

COMPACT ACTION. Instead of from(64) x to(64) = 4096 indices, about 99% of which
can never occur in draughts, a move is described by its FIRST step:

    index = dark_square(32) * 8 + direction(4) * 2 + mode(2)            -> 256

  direction in {NW, NE, SW, SE} in the CANONICAL FRAME (relative to the side to
  move: "we" advance towards row 0), mode in {step, jump}. Each direction owns
  two consecutive slots, step first and jump second: index = 8s + 2d + m.

  For a multiple capture the index describes the FIRST jump only. Two legal
  chains that share their first jump therefore fall on the SAME index: the
  collision is handled downstream by priors_from_logits (softmax over the
  distinct indices only, then the probability is split evenly among the moves
  sharing an index).

  256 is 16x smaller than 4096: fewer parameters, faster convergence, a lighter
  policy head. Light squares have no slots at all: they are not generated and
  then masked, they simply do not exist in the action space.
"""
from __future__ import annotations
import numpy as np

from .board import color_of, is_king, BLACK, N_SQUARES, DARK_SQUARES, DARK_INDEX
from .game import Position, NO_PROGRESS_DRAW
from .moves import Move, has_capture

IN_PLANES = 7
# Constant mask of the dark (playable) squares. The parity of (r+c) does not
# change under a 180-degree rotation, so it holds in the canonical frame too.
_DARK_MASK = np.array([[1.0 if (r + c) % 2 == 1 else 0.0 for c in range(8)]
                       for r in range(8)], dtype=np.float32)
N_DIRS = 4
N_MODES = 2
SLOTS = N_DIRS * N_MODES               # 8 slots per square
POLICY_SIZE = len(DARK_SQUARES) * SLOTS  # 32 * 8 = 256 (dark squares only)

# diagonal directions in a fixed order (canonical frame: "we" move up)
DIR_LIST = ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIR_TO_IDX = {d: i for i, d in enumerate(DIR_LIST)}


def _flip(pos: Position) -> bool:
    return pos.turn == BLACK


def canon_sq(s: int, flip: bool) -> int:
    return (N_SQUARES - 1 - s) if flip else s


def encode(pos: Position) -> np.ndarray:
    """Position -> [7, 8, 8] tensor, canonicalized to the point of view of the
    side to move. Planes: 0 our men, 1 our kings, 2 their men, 3 their kings,
    4 dark-square mask, 5 capture available (broadcast), 6 normalized
    no-progress counter (broadcast)."""
    planes = np.zeros((IN_PLANES, 8, 8), dtype=np.float32)
    flip = _flip(pos)
    turn = pos.turn
    for s in range(N_SQUARES):
        p = pos.board[s]
        if p == 0:
            continue
        cs = canon_sq(s, flip)
        r, c = divmod(cs, 8)
        rel = color_of(p) * turn        # +1 = ours, -1 = opponent's
        king = is_king(p)
        if rel == 1:
            planes[1 if king else 0, r, c] = 1.0
        else:
            planes[3 if king else 2, r, c] = 1.0
    # --- engineered planes ---
    planes[4] = _DARK_MASK                                    # geometry
    if has_capture(pos.board, turn):                         # tactics
        planes[5] = 1.0
    planes[6] = min(pos.no_progress / NO_PROGRESS_DRAW, 1.0)  # draw rule
    return planes


def move_policy_index(move: Move, flip: bool) -> int:
    """Compact action index (canonical frame): square*8 + direction*2 + mode,
    from the FIRST step of the move."""
    cf = canon_sq(move.frm, flip)
    c1 = canon_sq(move.path[1], flip)
    r0, col0 = divmod(cf, 8)
    r1, col1 = divmod(c1, 8)
    dr = (r1 > r0) - (r1 < r0)
    dc = (col1 > col0) - (col1 < col0)
    di = DIR_TO_IDX[(dr, dc)]
    mode = 1 if move.is_capture else 0
    return DARK_INDEX[cf] * SLOTS + di * N_MODES + mode   # compact index over 32 squares


def legal_actions(pos: Position):
    """Returns (moves, indices): the legal moves and their (compact, canonical)
    policy indices. This is the ACTION MASKING."""
    flip = _flip(pos)
    moves = pos.legal_moves()
    indices = [move_policy_index(m, flip) for m in moves]
    return moves, indices


def legal_mask(pos: Position) -> np.ndarray:
    """[POLICY_SIZE] vector with 1.0 on the legal actions, 0 elsewhere."""
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    _, indices = legal_actions(pos)
    for i in indices:
        mask[i] = 1.0
    return mask


def index_to_move(pos: Position, index: int) -> Move | None:
    """Inverse mapping: the legal Move for a policy index (or None). If several
    chains share the same first step -- rare after the capture-priority rules --
    the first one is returned (deterministic)."""
    moves, indices = legal_actions(pos)
    for m, i in zip(moves, indices):
        if i == index:
            return m
    return None
