"""
Italian draughts: legal move generation.

Rules implemented (ITALIAN draughts):
  - Man: moves and captures ONE square diagonally, FORWARD only.
  - King: moves and captures ONE square diagonally in all 4 directions
    (no flying kings: only the adjacent square).
  - Capturing is MANDATORY; a capture chain continues as long as possible.
  - A MAN cannot capture a KING (the distinctive Italian rule).
  - Promotion: a man reaching the last row becomes a king and the move ENDS
    (it does not go on capturing as a king).
  - Capture priority, applied in order:
      1. the largest number of pieces captured;
      2. if tied, capture with a KING rather than with a man;
      3. if tied, capture the largest number of KINGS;
      4. if tied, capture a king as early as possible in the chain.

A move is represented by `Move`; its policy index is computed by
encoder.move_policy_index.
"""
from __future__ import annotations
from dataclasses import dataclass

from .board import (
    EMPTY, N_SQUARES, DIRS_ALL, rc, sq, on_board, color_of, is_king, is_man,
    forward_dirs, promotion_row,
)

INF = 1 << 30


@dataclass(frozen=True)
class Move:
    frm: int                 # start square (0..63)
    to: int                  # final square (0..63)
    path: tuple              # full sequence of squares (frm, ..., to)
    captured: tuple          # captured squares, IN capture ORDER
    promotes: bool           # the man is promoted to king by this move
    by_king: bool            # the moving piece was already a king at the start

    @property
    def is_capture(self) -> bool:
        return len(self.captured) > 0


def _capture_dirs(piece: int, color: int):
    return DIRS_ALL if is_king(piece) else forward_dirs(color)


def _captures_from(board, start, piece, color, captured, path, out):
    """Recursively explores the capture chains starting from `start`.
    Jumped pieces stay on the board (they block) but are listed in `captured`,
    so they cannot be jumped twice. Only MAXIMAL chains are recorded in `out`."""
    r0, c0 = rc(start)
    extended = False
    for dr, dc in _capture_dirs(piece, color):
        rm, cm = r0 + dr, c0 + dc          # square of the piece to jump
        rl, cl = r0 + 2 * dr, c0 + 2 * dc  # landing square
        if not on_board(rl, cl):
            continue
        mid, land = sq(rm, cm), sq(rl, cl)
        target = board[mid]
        # the jumped piece must be an enemy not captured yet
        if color_of(target) != -color or mid in captured:
            continue
        # a MAN does not capture a KING
        if is_man(piece) and is_king(target):
            continue
        # the landing square must be empty (captured pieces stay on the board)
        if board[land] != EMPTY or land in captured:
            continue

        new_captured = captured + (mid,)
        new_path = path + (land,)
        # promotion in the middle of a chain: the man is promoted and the move ENDS
        if is_man(piece) and rl == promotion_row(color):
            out.append((new_path, new_captured, True))
            extended = True
            continue
        extended = True
        _captures_from(board, land, piece, color, new_captured, new_path, out)

    # no possible continuation and at least one capture made -> the chain is complete
    if not extended and len(captured) > 0:
        out.append((path, captured, False))


def _gen_captures(board, color):
    """All the maximal capture chains for the given color."""
    moves = []
    for s in range(N_SQUARES):
        piece = board[s]
        if color_of(piece) != color:
            continue
        # The moving piece is "in hand": its origin is cleared so that a cyclic
        # chain (king) can pass through it again; jumped squares stay occupied.
        work = list(board)
        work[s] = EMPTY
        raw = []
        _captures_from(work, s, piece, color, tuple(), (s,), raw)
        by_king = is_king(piece)
        for path, captured, promotes in raw:
            moves.append(Move(frm=s, to=path[-1], path=path, captured=captured,
                              promotes=promotes, by_king=by_king))
    return moves


def _gen_simple(board, color):
    """Simple (non-capturing) moves: one square diagonally."""
    moves = []
    for s in range(N_SQUARES):
        piece = board[s]
        if color_of(piece) != color:
            continue
        r0, c0 = rc(s)
        dirs = DIRS_ALL if is_king(piece) else forward_dirs(color)
        for dr, dc in dirs:
            r1, c1 = r0 + dr, c0 + dc
            if not on_board(r1, c1):
                continue
            d = sq(r1, c1)
            if board[d] != EMPTY:
                continue
            promotes = is_man(piece) and r1 == promotion_row(color)
            moves.append(Move(frm=s, to=d, path=(s, d), captured=tuple(),
                              promotes=promotes, by_king=is_king(piece)))
    return moves


def _kings_captured(board, captured):
    return sum(1 for m in captured if is_king(board[m]))


def _first_king_step(board, captured):
    for i, m in enumerate(captured):
        if is_king(board[m]):
            return i
    return INF


def _apply_priority(board, captures):
    """Filters the captures by the Italian capture-priority rules (1-4)."""
    mx = max(len(m.captured) for m in captures)
    c = [m for m in captures if len(m.captured) == mx]          # 1: most pieces
    if any(m.by_king for m in c):                              # 2: with a king
        c = [m for m in c if m.by_king]
    mk = max(_kings_captured(board, m.captured) for m in c)     # 3: most kings
    c = [m for m in c if _kings_captured(board, m.captured) == mk]
    if mk > 0:                                                  # 4: king earliest
        fk = min(_first_king_step(board, m.captured) for m in c)
        c = [m for m in c if _first_king_step(board, m.captured) == fk]
    return c


def generate_legal_moves(board, color):
    """Legal moves for `color`. If there are captures they are mandatory and
    filtered by capture priority; otherwise the simple moves."""
    captures = _gen_captures(board, color)
    if captures:
        return _apply_priority(board, captures)
    return _gen_simple(board, color)


def apply_move(board, move: Move, color: int) -> list[int]:
    """Returns a NEW board after playing `move`."""
    nb = list(board)
    piece = nb[move.frm]
    nb[move.frm] = EMPTY
    for m in move.captured:
        nb[m] = EMPTY
    # promotion to king (2 with the sign of the color), otherwise unchanged
    nb[move.to] = (2 * color) if move.promotes else piece
    return nb


def has_capture(board, color) -> bool:
    """True if `color` has at least one capture available (captures are
    mandatory). Checks a single jump per piece: enough for the input plane, and
    much cheaper than generating every chain."""
    for s in range(N_SQUARES):
        piece = board[s]
        if color_of(piece) != color:
            continue
        r0, c0 = rc(s)
        for dr, dc in _capture_dirs(piece, color):
            rl, cl = r0 + 2 * dr, c0 + 2 * dc
            if not on_board(rl, cl):
                continue
            target = board[sq(r0 + dr, c0 + dc)]
            if color_of(target) != -color:
                continue
            if is_man(piece) and is_king(target):
                continue
            if board[sq(rl, cl)] != EMPTY:
                continue
            return True
    return False
