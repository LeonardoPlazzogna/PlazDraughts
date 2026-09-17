/**
 * Square numbers and game text, in the project's own format.
 *
 * The dark squares are numbered 1..32 from the top left, exactly as play.py
 * numbers them, and a game is written the way tools/kingsrow.py writes one --
 * so a game played in the browser can be read back by tools/read_games.py and
 * measured by tools/quiet_material.py without a single new format to support.
 */
import { N_SQUARES, WHITE } from "../engine/board.js";

/** square 0..63 -> 1..32 on the dark squares, 0 elsewhere. */
export const SQUARE_NUMBER = new Int8Array(N_SQUARES);
{
  let k = 0;
  for (let s = 0; s < N_SQUARES; s++) {
    if (((s >> 3) + (s & 7)) % 2 === 1) SQUARE_NUMBER[s] = ++k;
  }
}

/** "22-18", "18x11", "26x17x10x1=D": the path, and =D when a man is promoted. */
export function moveText(move) {
  const sep = move.isCapture ? "x" : "-";
  return move.path.map((s) => SQUARE_NUMBER[s]).join(sep) + (move.promotes ? "=D" : "");
}

/**
 * The whole game as text, in the format of results/games/.
 *
 * `moves` is the list of moves in order, starting with White's.
 */
export function gameText({ moves, humanIsWhite, outcome, reason, model, sims }) {
  const rule = "=".repeat(68);
  const lines = [
    `${model} vs a human player`,
    `engine: ${sims} simulations per move`,
    "dark squares numbered 1..32; x = capture, =D = promotion",
    "",
    rule,
    `Game 1  --  we play ${humanIsWhite ? "Black" : "White"}`
      + `  --  ${outcome.toUpperCase()}  (${moves.length} plies, ${reason})`,
    rule,
  ];
  for (let i = 0; i < moves.length; i += 2) {
    const a = moveText(moves[i]);
    const b = moves[i + 1] ? moveText(moves[i + 1]) : "";
    lines.push(`${String(i / 2 + 1).padStart(3)}. ${a.padEnd(16)} ${b}`);
  }
  lines.push("");
  return lines.join("\n");
}

/**
 * The outcome as the game files spell it, from the ENGINE's point of view --
 * "we" in that format is always the program, so a human win is a loss there.
 */
export function outcomeFor(result, humanIsWhite) {
  if (result === 0) return "draw";
  const humanWon = (result === WHITE) === humanIsWhite;
  return humanWon ? "loss" : "win";
}
