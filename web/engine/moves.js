/**
 * Italian draughts: legal move generation. Port of dama/moves.py.
 *
 * The rules, in the order they bite:
 *   - a man moves and captures ONE square diagonally, forward only;
 *   - a king does the same in all four directions (no flying kings);
 *   - capturing is MANDATORY and a chain continues while it can;
 *   - a MAN cannot capture a KING -- the distinctive Italian rule;
 *   - a man promoted mid-chain stops there: the move ends on promotion;
 *   - among the available captures, priority decides, in order:
 *       1. the largest number of pieces;
 *       2. if tied, capture with a KING rather than with a man;
 *       3. if tied, capture the largest number of kings;
 *       4. if tied, meet a king as early as possible in the chain.
 *
 * Rules 2 to 4 are what separate Italian draughts from every other variant, and
 * they are the reason this file is ported literally instead of rewritten: a
 * board that looks right while applying English rules produces normal-looking
 * games that are not the same game.
 */
import {
  EMPTY, N_SQUARES, DIRS_ALL, rowOf, colOf, sq, onBoard, colorOf, isKing, isMan,
  forwardDirs, promotionRow,
} from "./board.js";

const INF = 1 << 30;

/** A move: the full path, the captured squares IN capture order, and the flags. */
export function makeMove(frm, to, path, captured, promotes, byKing) {
  return { frm, to, path, captured, promotes, byKing, isCapture: captured.length > 0 };
}

const captureDirs = (piece, color) => (isKing(piece) ? DIRS_ALL : forwardDirs(color));

/**
 * Explores the capture chains from `start`, recording only the MAXIMAL ones.
 *
 * Jumped pieces stay on the board -- they still block the squares they stand on
 * -- but are listed in `captured` so they cannot be jumped twice. The piece in
 * hand has already been lifted from its origin by the caller, so a king's chain
 * can pass through that square again.
 */
function capturesFrom(board, start, piece, color, captured, path, out) {
  const r0 = rowOf(start), c0 = colOf(start);
  let extended = false;
  for (const [dr, dc] of captureDirs(piece, color)) {
    const rm = r0 + dr, cm = c0 + dc;
    const rl = r0 + 2 * dr, cl = c0 + 2 * dc;
    if (!onBoard(rl, cl)) continue;
    const mid = sq(rm, cm), land = sq(rl, cl);
    const target = board[mid];
    if (colorOf(target) !== -color || captured.includes(mid)) continue;
    if (isMan(piece) && isKing(target)) continue;              // a man spares kings
    if (board[land] !== EMPTY || captured.includes(land)) continue;

    const newCaptured = captured.concat(mid);
    const newPath = path.concat(land);
    if (isMan(piece) && rl === promotionRow(color)) {
      out.push({ path: newPath, captured: newCaptured, promotes: true });
      extended = true;
      continue;
    }
    extended = true;
    capturesFrom(board, land, piece, color, newCaptured, newPath, out);
  }
  if (!extended && captured.length > 0) {
    out.push({ path, captured, promotes: false });
  }
}

function genCaptures(board, color) {
  const moves = [];
  for (let s = 0; s < N_SQUARES; s++) {
    const piece = board[s];
    if (colorOf(piece) !== color) continue;
    const work = Int8Array.from(board);
    work[s] = EMPTY;                       // the piece is in hand
    const raw = [];
    capturesFrom(work, s, piece, color, [], [s], raw);
    const byKing = isKing(piece);
    for (const { path, captured, promotes } of raw) {
      moves.push(makeMove(s, path[path.length - 1], path, captured, promotes, byKing));
    }
  }
  return moves;
}

function genSimple(board, color) {
  const moves = [];
  for (let s = 0; s < N_SQUARES; s++) {
    const piece = board[s];
    if (colorOf(piece) !== color) continue;
    const r0 = rowOf(s), c0 = colOf(s);
    const dirs = isKing(piece) ? DIRS_ALL : forwardDirs(color);
    for (const [dr, dc] of dirs) {
      const r1 = r0 + dr, c1 = c0 + dc;
      if (!onBoard(r1, c1)) continue;
      const d = sq(r1, c1);
      if (board[d] !== EMPTY) continue;
      const promotes = isMan(piece) && r1 === promotionRow(color);
      moves.push(makeMove(s, d, [s, d], [], promotes, isKing(piece)));
    }
  }
  return moves;
}

const kingsCaptured = (board, captured) =>
  captured.reduce((n, m) => n + (isKing(board[m]) ? 1 : 0), 0);

function firstKingStep(board, captured) {
  for (let i = 0; i < captured.length; i++) if (isKing(board[captured[i]])) return i;
  return INF;
}

/** Filters the captures by the Italian priority rules (1 to 4). */
function applyPriority(board, captures) {
  const mx = Math.max(...captures.map((m) => m.captured.length));
  let c = captures.filter((m) => m.captured.length === mx);
  if (c.some((m) => m.byKing)) c = c.filter((m) => m.byKing);
  const mk = Math.max(...c.map((m) => kingsCaptured(board, m.captured)));
  c = c.filter((m) => kingsCaptured(board, m.captured) === mk);
  if (mk > 0) {
    const fk = Math.min(...c.map((m) => firstKingStep(board, m.captured)));
    c = c.filter((m) => firstKingStep(board, m.captured) === fk);
  }
  return c;
}

/** Legal moves for `color`: the captures if there are any, otherwise the simple moves. */
export function generateLegalMoves(board, color) {
  const captures = genCaptures(board, color);
  if (captures.length > 0) return applyPriority(board, captures);
  return genSimple(board, color);
}

/** A NEW board after playing `move`. */
export function applyMove(board, move, color) {
  const nb = Int8Array.from(board);
  const piece = nb[move.frm];
  nb[move.frm] = EMPTY;
  for (const m of move.captured) nb[m] = EMPTY;
  nb[move.to] = move.promotes ? 2 * color : piece;
  return nb;
}

/**
 * True if `color` has at least one capture available.
 *
 * One jump per piece is enough here -- the answer is a yes/no plane for the
 * network, not a move list -- and it costs a fraction of generating the chains.
 */
export function hasCapture(board, color) {
  for (let s = 0; s < N_SQUARES; s++) {
    const piece = board[s];
    if (colorOf(piece) !== color) continue;
    const r0 = rowOf(s), c0 = colOf(s);
    for (const [dr, dc] of captureDirs(piece, color)) {
      const rl = r0 + 2 * dr, cl = c0 + 2 * dc;
      if (!onBoard(rl, cl)) continue;
      const target = board[sq(r0 + dr, c0 + dc)];
      if (colorOf(target) !== -color) continue;
      if (isMan(piece) && isKing(target)) continue;
      if (board[sq(rl, cl)] !== EMPTY) continue;
      return true;
    }
  }
  return false;
}
