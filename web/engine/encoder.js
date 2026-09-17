/**
 * Italian draughts: position -> network input, and the compact action space.
 * Port of dama/encoder.py.
 *
 * INPUT: a [7, 8, 8] tensor, flattened in the same order numpy uses
 * (plane*64 + row*8 + col), canonicalized to the side to move: for Black the
 * board is rotated 180 degrees and the colors swapped, so square s becomes
 * 63 - s and the network always sees itself moving up. Planes: 0 our men,
 * 1 our kings, 2 their men, 3 their kings, 4 dark-square mask, 5 capture
 * available, 6 the no-progress counter, normalized.
 *
 * ACTION: a move is named by its FIRST step,
 *
 *     index = dark_square(32) * 8 + direction(4) * 2 + mode(2)   -> 256
 *
 * with the directions in the canonical frame and mode in {step, jump}. Two
 * capture chains that share their first jump land on the SAME index; the
 * collision is resolved where the priors are built, by splitting the
 * probability evenly among the moves that share an index.
 */
import { colorOf, isKing, BLACK, N_SQUARES, DARK_SQUARES, DARK_INDEX } from "./board.js";
import { NO_PROGRESS_DRAW } from "./game.js";
import { hasCapture } from "./moves.js";

export const IN_PLANES = 7;
export const N_DIRS = 4;
export const N_MODES = 2;
export const SLOTS = N_DIRS * N_MODES;
export const POLICY_SIZE = DARK_SQUARES.length * SLOTS;   // 32 * 8 = 256
export const PLANE_SIZE = 64;

/**
 * Direction index in the canonical frame, in the same fixed order the Python
 * encoder uses: (-1,-1), (-1,+1), (+1,-1), (+1,+1). The order is not a detail:
 * it is baked into the trained weights, so a different one would silently map
 * every move to the wrong slot.
 */
const dirIndex = (dr, dc) => (dr < 0 ? (dc < 0 ? 0 : 1) : (dc < 0 ? 2 : 3));

const flipOf = (pos) => pos.turn === BLACK;
export const canonSq = (s, flip) => (flip ? N_SQUARES - 1 - s : s);

/** Position -> Float32Array(7*8*8), ready for the network. */
export function encode(pos) {
  const planes = new Float32Array(IN_PLANES * PLANE_SIZE);
  const flip = flipOf(pos);
  const turn = pos.turn;
  for (let s = 0; s < N_SQUARES; s++) {
    const p = pos.board[s];
    if (p === 0) continue;
    const cs = canonSq(s, flip);
    const rel = colorOf(p) * turn;               // +1 ours, -1 theirs
    const king = isKing(p);
    const plane = rel === 1 ? (king ? 1 : 0) : (king ? 3 : 2);
    planes[plane * PLANE_SIZE + cs] = 1.0;
  }
  // The dark-square mask: the parity of (row + col) survives a 180-degree
  // rotation, so the same mask holds in the canonical frame.
  for (let s = 0; s < N_SQUARES; s++) {
    if (((s >> 3) + (s & 7)) % 2 === 1) planes[4 * PLANE_SIZE + s] = 1.0;
  }
  // Captures are mandatory, so whether one exists changes what every move means:
  // the plane is broadcast over the whole board, like the counter below.
  if (hasCapture(pos.board, turn)) planes.fill(1.0, 5 * PLANE_SIZE, 6 * PLANE_SIZE);
  const np = Math.min(pos.noProgress / NO_PROGRESS_DRAW, 1.0);
  planes.fill(np, 6 * PLANE_SIZE, 7 * PLANE_SIZE);
  return planes;
}

/** Compact action index of a move, in the canonical frame. */
export function movePolicyIndex(move, flip) {
  const cf = canonSq(move.frm, flip);
  const c1 = canonSq(move.path[1], flip);
  const r0 = cf >> 3, col0 = cf & 7;
  const r1 = c1 >> 3, col1 = c1 & 7;
  const dr = Math.sign(r1 - r0);
  const dc = Math.sign(col1 - col0);
  const di = dirIndex(dr, dc);
  const mode = move.isCapture ? 1 : 0;
  return DARK_INDEX[cf] * SLOTS + di * N_MODES + mode;
}

/** The legal moves and their policy indices: this is the action masking. */
export function legalActions(pos) {
  const flip = flipOf(pos);
  const moves = pos.legalMoves();
  return { moves, indices: moves.map((m) => movePolicyIndex(m, flip)) };
}

/**
 * Raw policy logits -> probabilities over the legal moves only.
 *
 * Port of evaluators.priors_from_logits, collisions included: the softmax runs
 * over the DISTINCT indices, so a logit shared by several chains is not counted
 * twice, and the probability of an index is then split evenly among them. The
 * network cannot tell those chains apart, so an even split is the only
 * principled choice -- and the priors still sum to 1.
 */
export function priorsFromLogits(pos, logits) {
  const { moves, indices } = legalActions(pos);
  if (moves.length === 0) return [];
  const uniq = [];
  for (const ix of indices) if (!uniq.includes(ix)) uniq.push(ix);
  let max = -Infinity;
  for (const ix of uniq) if (logits[ix] > max) max = logits[ix];
  let sum = 0;
  const exp = uniq.map((ix) => { const e = Math.exp(logits[ix] - max); sum += e; return e; });
  const probOf = new Map();
  uniq.forEach((ix, j) => probOf.set(ix, exp[j] / sum));
  const counts = new Map();
  for (const ix of indices) counts.set(ix, (counts.get(ix) ?? 0) + 1);
  return moves.map((m, j) => ({
    move: m,
    prior: probOf.get(indices[j]) / counts.get(indices[j]),
  }));
}

/** The legal move for a policy index, or null. */
export function indexToMove(pos, index) {
  const { moves, indices } = legalActions(pos);
  for (let i = 0; i < moves.length; i++) if (indices[i] === index) return moves[i];
  return null;
}
