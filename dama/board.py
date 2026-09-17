"""
Italian draughts: board representation and constants.

8x8 board, squares indexed 0..63 (sq = row*8 + col), row 0 at the TOP.
Only the dark squares are played on: those where (row + col) is odd.

Colors and directions:
  - WHITE (+1) starts at the bottom (rows 5, 6, 7); men move up -> dr = -1.
    Promotion on row 0.
  - BLACK (-1) starts at the top (rows 0, 1, 2); men move down -> dr = +1.
    Promotion on row 7.

Piece encoding (signed integer: sign = color, magnitude = type):
   0  empty
  +1  white man      +2  white king
  -1  black man      -2  black king

The board is a plain list of 64 ints, with two piece types only: man and king.
"""
from __future__ import annotations

# --- pieces ---
EMPTY = 0
W_MAN, W_KING = 1, 2
B_MAN, B_KING = -1, -2

# --- colors ---
WHITE, BLACK = 1, -1

BOARD_SIZE = 8
N_SQUARES = BOARD_SIZE * BOARD_SIZE  # 64

# The 4 diagonals (dr, dc).
DIRS_ALL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def rc(sq: int) -> tuple[int, int]:
    """From index 0..63 to (row, column)."""
    return divmod(sq, BOARD_SIZE)


def sq(r: int, c: int) -> int:
    """From (row, column) to index 0..63."""
    return r * BOARD_SIZE + c


def on_board(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


# The 32 dark (playable) squares in index order, and the map square -> compact
# index 0..31. Used by the compact 32*8 = 256 policy, which has no outputs on
# the light squares, where no move is ever legal.
DARK_SQUARES = tuple(s for s in range(N_SQUARES) if (s // 8 + s % 8) % 2 == 1)
DARK_INDEX = {s: i for i, s in enumerate(DARK_SQUARES)}


def color_of(piece: int) -> int:
    """+1 if white, -1 if black, 0 if empty."""
    return (piece > 0) - (piece < 0)


def is_king(piece: int) -> bool:
    return abs(piece) == 2


def is_man(piece: int) -> bool:
    return abs(piece) == 1


def forward_dirs(color: int) -> tuple[tuple[int, int], ...]:
    """Directions in which the MEN of the given color move."""
    if color == WHITE:
        return ((-1, -1), (-1, 1))   # towards row 0
    return ((1, -1), (1, 1))         # towards row 7


def promotion_row(color: int) -> int:
    return 0 if color == WHITE else BOARD_SIZE - 1


def initial_board() -> list[int]:
    """Starting position: 12 men per side on the dark squares of the 3 outer
    rows. White at the bottom (5, 6, 7), Black at the top (0, 1, 2)."""
    b = [EMPTY] * N_SQUARES
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r + c) % 2 != 1:
                continue  # dark squares only
            if r <= 2:
                b[sq(r, c)] = B_MAN
            elif r >= 5:
                b[sq(r, c)] = W_MAN
    return b


_GLYPH = {EMPTY: ".", W_MAN: "w", W_KING: "W", B_MAN: "b", B_KING: "B"}


def render(board: list[int]) -> str:
    """Text rendering (upper case = kings). Row 0 at the top."""
    lines = []
    for r in range(BOARD_SIZE):
        row = " ".join(_GLYPH[board[sq(r, c)]] for c in range(BOARD_SIZE))
        lines.append(f"{r} {row}")
    lines.append("  " + " ".join(str(c) for c in range(BOARD_SIZE)))
    return "\n".join(lines)
