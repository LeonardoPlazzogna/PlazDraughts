/**
 * The game text the page saves must be the text the project already reads.
 *
 * tools/read_games.py parses the files in results/games/ with one regular
 * expression, and tools/quiet_material.py builds the thesis figures on top of
 * it. A game saved from the browser is worth something only if those two can
 * open it, so the header is checked against THAT regular expression, copied
 * here, and the moves against the notation of a real recorded game.
 *
 * Run: node --test web/test/notation.test.js
 */
import test from "node:test";
import assert from "node:assert/strict";

import { W_MAN, sq } from "../engine/board.js";
import { makeMove } from "../engine/moves.js";
import { SQUARE_NUMBER, moveText, gameText, outcomeFor } from "../app/notation.js";

// tools/read_games.py:
//   r"Game (\d+)\s+--\s+we play (\w+)\s+--\s+(\w+)\s+\((\d+) plies, ([^)]+)\)"
const HEADER = /Game (\d+)\s+--\s+we play (\w+)\s+--\s+(\w+)\s+\((\d+) plies, ([^)]+)\)/;

test("the dark squares are numbered 1..32 as play.py numbers them", () => {
  assert.equal(SQUARE_NUMBER[sq(0, 1)], 1);
  assert.equal(SQUARE_NUMBER[sq(0, 7)], 4);
  assert.equal(SQUARE_NUMBER[sq(4, 3)], 18);      // the square of "22-18"
  assert.equal(SQUARE_NUMBER[sq(5, 2)], 22);
  assert.equal(SQUARE_NUMBER[sq(7, 6)], 32);
  // A light square has no number at all: no move ever lands there.
  assert.equal(SQUARE_NUMBER[sq(0, 0)], 0);
});

test("a quiet move, a capture and a promotion read as the recorded games do", () => {
  const quiet = makeMove(sq(5, 2), sq(4, 3), [sq(5, 2), sq(4, 3)], [], false, false);
  assert.equal(moveText(quiet), "22-18");

  const capture = makeMove(sq(4, 3), sq(2, 5), [sq(4, 3), sq(2, 5)], [sq(3, 4)], false, false);
  assert.equal(moveText(capture), "18x11");

  // A chain that ends on the last row: the path, then =D.
  const chain = makeMove(sq(6, 1), sq(0, 3),
    [sq(6, 1), sq(4, 3), sq(2, 5), sq(0, 3)],
    [sq(5, 2), sq(3, 4), sq(1, 4)], true, false);
  assert.equal(moveText(chain), "25x18x11x2=D");
});

test("the header is the one tools/read_games.py knows how to read", () => {
  const moves = [
    makeMove(sq(5, 2), sq(4, 3), [sq(5, 2), sq(4, 3)], [], false, false),
    makeMove(sq(2, 5), sq(3, 4), [sq(2, 5), sq(3, 4)], [], false, false),
    makeMove(sq(4, 3), sq(2, 5), [sq(4, 3), sq(2, 5)], [sq(3, 4)], false, false),
  ];
  const text = gameText({
    moves, humanIsWhite: true, outcome: "loss", reason: "no legal moves",
    model: "run B, cycle 200", sims: 400,
  });

  const m = HEADER.exec(text);
  assert.ok(m, "the header must match the reader's regular expression");
  assert.equal(m[1], "1");
  // "we" in those files is always the program, so with the human as White the
  // engine is Black.
  assert.equal(m[2], "Black");
  assert.equal(m[3], "LOSS");
  assert.equal(m[4], String(moves.length));
  assert.equal(m[5], "no legal moves");

  const lines = text.split("\n");
  assert.ok(lines.some((l) => /^\s{2}1\. 22-18\s+11-15$/.test(l)),
    "the moves go in numbered pairs, White first");
  assert.ok(lines.some((l) => /^\s{2}2\. 18x11\s*$/.test(l)),
    "an unanswered move leaves the second column empty");
});

test("the outcome is written from the engine's side, as the files do", () => {
  assert.equal(outcomeFor(0, true), "draw");
  assert.equal(outcomeFor(1, true), "loss");    // White wins and the human is White
  assert.equal(outcomeFor(-1, true), "win");    // Black wins, and Black is the engine
  assert.equal(outcomeFor(-1, false), "loss");  // the human plays Black and wins
});
