/**
 * Italian draughts: board representation and constants.
 *
 * Port of dama/board.py, kept deliberately literal: same names, same encoding,
 * same square numbering. The browser has to play the game the Python engine
 * plays, and the cheapest way to keep a port honest is to make it and its
 * original easy to read side by side.
 *
 * 8x8 board, squares 0..63 (sq = row*8 + col), row 0 at the TOP. Only the dark
 * squares, where (row + col) is odd, are played on.
 *
 * Pieces: sign = color, magnitude = type. 0 empty, +-1 man, +-2 king.
 * WHITE (+1) starts at the bottom and its men move towards row 0.
 */

export const EMPTY = 0;
export const W_MAN = 1, W_KING = 2;
export const B_MAN = -1, B_KING = -2;

export const WHITE = 1, BLACK = -1;

export const BOARD_SIZE = 8;
export const N_SQUARES = BOARD_SIZE * BOARD_SIZE;

/** The 4 diagonals (dr, dc). */
export const DIRS_ALL = [[-1, -1], [-1, 1], [1, -1], [1, 1]];

export const rowOf = (s) => (s / BOARD_SIZE) | 0;
export const colOf = (s) => s % BOARD_SIZE;
export const sq = (r, c) => r * BOARD_SIZE + c;
export const onBoard = (r, c) => r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;

/**
 * The 32 dark squares in index order, and square -> compact index 0..31.
 * The compact 32*8 = 256 policy has no outputs on the light squares, where no
 * move can ever be legal.
 */
export const DARK_SQUARES = [];
for (let s = 0; s < N_SQUARES; s++) {
  if ((rowOf(s) + colOf(s)) % 2 === 1) DARK_SQUARES.push(s);
}
export const DARK_INDEX = new Int8Array(N_SQUARES).fill(-1);
DARK_SQUARES.forEach((s, i) => { DARK_INDEX[s] = i; });

/** +1 white, -1 black, 0 empty. */
export const colorOf = (piece) => (piece > 0 ? 1 : piece < 0 ? -1 : 0);
export const isKing = (piece) => piece === 2 || piece === -2;
export const isMan = (piece) => piece === 1 || piece === -1;

/** Directions in which the MEN of the given color move. */
export function forwardDirs(color) {
  return color === WHITE ? [[-1, -1], [-1, 1]] : [[1, -1], [1, 1]];
}

export const promotionRow = (color) => (color === WHITE ? 0 : BOARD_SIZE - 1);

/** Starting position: 12 men a side on the dark squares of the three outer rows. */
export function initialBoard() {
  const b = new Int8Array(N_SQUARES);
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if ((r + c) % 2 !== 1) continue;
      if (r <= 2) b[sq(r, c)] = B_MAN;
      else if (r >= 5) b[sq(r, c)] = W_MAN;
    }
  }
  return b;
}

const GLYPH = { 0: ".", 1: "w", 2: "W", [-1]: "b", [-2]: "B" };

/**
 * Text rendering (upper case = kings), row 0 at the top.
 *
 * Nothing in the page draws with it -- the board is SVG. It exists so that a
 * position can be printed while poking at the engine from a browser console,
 * which is where a port gets debugged.
 */
export function render(board) {
  const lines = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    const row = [];
    for (let c = 0; c < BOARD_SIZE; c++) row.push(GLYPH[board[sq(r, c)]]);
    lines.push(`${r} ${row.join(" ")}`);
  }
  lines.push("  " + [0, 1, 2, 3, 4, 5, 6, 7].join(" "));
  return lines.join("\n");
}
