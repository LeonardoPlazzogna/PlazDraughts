/**
 * Italian draughts: game state, transitions and outcome. Port of dama/game.py.
 *
 * A Position bundles the board, the side to move and the no-progress counter,
 * and never changes once built: `play` returns a new one. The search relies on
 * that, and so does the memoized move list.
 */
import { WHITE, initialBoard, render } from "./board.js";
import { generateLegalMoves, applyMove } from "./moves.js";

/**
 * Plies without a capture or a promotion after which the game is drawn.
 *
 * NOT the official rule, and knowingly so: the FID rules count 40 moves of the
 * side being counted, with at least one king a side and on a player's request,
 * and above all they reset the count on every MAN move. Here the count resets
 * only on a capture or a promotion, so games end EARLIER than the official rule
 * allows. It is left as it is because changing it changes the game: every
 * measurement already published used this criterion, and the C++ engine and the
 * Python one would have to change together. docs/results.md, section 9.
 */
export const NO_PROGRESS_DRAW = 80;

export class Position {
  constructor(board = null, turn = WHITE, noProgress = 0) {
    this.board = board ?? initialBoard();
    this.turn = turn;
    this.noProgress = noProgress;
    this._legal = null;
  }

  /** The legal moves, generated once: several callers ask for the same list. */
  legalMoves() {
    if (this._legal === null) this._legal = generateLegalMoves(this.board, this.turn);
    return this._legal;
  }

  /** The position after `move`. This one is not modified. */
  play(move) {
    const nb = applyMove(this.board, move, this.turn);
    const progressed = move.isCapture || move.promotes;
    return new Position(nb, -this.turn, progressed ? 0 : this.noProgress + 1);
  }

  isTerminal() {
    if (this.noProgress >= NO_PROGRESS_DRAW) return true;
    return this.legalMoves().length === 0;
  }

  /**
   * Outcome from WHITE's point of view: +1, -1, 0, or null if the game is on.
   * The side to move with no legal moves has lost -- no pieces, or all blocked.
   */
  result() {
    if (this.noProgress >= NO_PROGRESS_DRAW) return 0;
    if (this.legalMoves().length === 0) return -this.turn;
    return null;
  }

  /** Terminal value seen from `perspective` (+1 white, -1 black). */
  terminalValue(perspective) {
    const r = this.result();
    return r === null ? 0 : r * perspective;
  }

  /** Hashable key (board + side to move), for transposition tables. */
  key() {
    return `${this.board.join(",")}|${this.turn}`;
  }

  toString() {
    const side = this.turn === WHITE ? "White" : "Black";
    return `${render(this.board)}\nTo move: ${side}  no_progress=${this.noProgress}`;
  }
}
