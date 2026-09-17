/**
 * The browser rules against the references the project already trusts.
 *
 * The risk of a third implementation of the rules -- after Python and C++ -- is
 * not that it crashes: it is that it plays a game that looks like Italian
 * draughts and is not. So this file does not check the port against itself. It
 * checks it against numbers produced elsewhere:
 *
 *   - the perft counts of engine_c/perft.cpp, which the C++ engine and the
 *     Python generator already agree on down to the last node;
 *   - the five positions of tools/rules_crosscheck.py, each isolating one of
 *     the Italian capture-priority rules, and verified against an external
 *     engine.
 *
 * Run: node --test web/test/rules.test.js
 */
import test from "node:test";
import assert from "node:assert/strict";

import { WHITE, W_MAN, W_KING, B_MAN, B_KING, sq, N_SQUARES } from "../engine/board.js";
import { Position } from "../engine/game.js";

/** Leaf nodes at depth `d`, the same definition engine_c/perft.cpp uses. */
function perft(pos, d) {
  if (d === 0) return 1;
  const moves = pos.legalMoves();
  if (moves.length === 0) return 0;
  let n = 0;
  for (const m of moves) n += perft(pos.play(m), d - 1);
  return n;
}

test("perft matches the reference counts of the C++ engine", () => {
  const expected = [1, 7, 49, 302, 1469, 7361, 36473, 177532];
  for (let d = 0; d < expected.length; d++) {
    assert.equal(perft(new Position(), d), expected[d], `perft(${d})`);
  }
});

/** A position from a {"row,col": piece} map, White to move, as the crosscheck builds it. */
function build(pieces) {
  const b = new Int8Array(N_SQUARES);
  for (const [key, piece] of Object.entries(pieces)) {
    const [r, c] = key.split(",").map(Number);
    assert.equal((r + c) % 2, 1, `light square: ${key}`);
    b[sq(r, c)] = piece;
  }
  return new Position(b, WHITE, 0);
}

/** The capture chains as sorted lists of captured squares, to compare by value. */
const captureSets = (pos) =>
  pos.legalMoves().map((m) => m.captured.slice().sort((a, b) => a - b).join(","));

test("priority 1: the longer chain is compulsory", () => {
  const pos = build({
    "5,4": W_MAN,
    "4,3": B_MAN, "2,1": B_MAN,   // two-piece branch
    "4,5": B_MAN,                  // one-piece branch
  });
  const moves = pos.legalMoves();
  assert.equal(moves.length, 1);
  assert.equal(moves[0].captured.length, 2);
  assert.deepEqual(captureSets(pos), [[sq(4, 3), sq(2, 1)].sort((a, b) => a - b).join(",")]);
});

test("priority 2: with equal numbers, the king captures", () => {
  const pos = build({
    "5,0": W_KING, "5,4": W_MAN,
    "4,1": B_MAN,                  // the king's prey
    "4,3": B_MAN,                  // the man's prey
  });
  const moves = pos.legalMoves();
  assert.equal(moves.length, 1);
  assert.ok(moves[0].byKing, "the move must be the king's");
  assert.deepEqual(moves[0].captured, [sq(4, 1)]);
});

test("priority 3: with equal numbers and the same piece, more kings win", () => {
  const pos = build({
    "4,3": W_KING,
    "3,2": B_KING,                 // prey: a king
    "3,4": B_MAN,                  // prey: a man
  });
  const moves = pos.legalMoves();
  assert.equal(moves.length, 1);
  assert.deepEqual(moves[0].captured, [sq(3, 2)]);
});

test("priority 4: the king met earliest wins", () => {
  const pos = build({
    "4,3": W_KING,
    "3,2": B_KING, "1,2": B_MAN,   // king at the FIRST jump
    "5,4": B_MAN, "5,6": B_KING,   // king at the SECOND jump
  });
  const moves = pos.legalMoves();
  assert.equal(moves.length, 1);
  assert.equal(moves[0].captured[0], sq(3, 2), "the first captured piece must be the king");
});

test("a man does not capture a king: the king is not even an option", () => {
  const pos = build({
    "5,2": W_MAN,
    "4,1": B_KING,                 // untouchable by a man
    "4,3": B_MAN,                  // the only possible prey
  });
  const moves = pos.legalMoves();
  assert.equal(moves.length, 1);
  assert.deepEqual(moves[0].captured, [sq(4, 3)]);
});

test("a promoted man stops: the chain ends on promotion", () => {
  // A white man one jump away from row 0, with a second capture waiting beyond
  // the promotion square. Italian rules end the move at the promotion.
  const pos = build({
    "2,3": W_MAN,
    "1,2": B_MAN,                  // jumped, landing on row 0 -> promotion
    "1,6": B_MAN, "2,7": B_MAN,    // bait: a chain that must NOT be continued
  });
  const moves = pos.legalMoves();
  assert.equal(moves.length, 1);
  assert.equal(moves[0].captured.length, 1);
  assert.ok(moves[0].promotes);
  assert.equal(moves[0].to, sq(0, 1));
});

test("the side to move with no legal moves has lost", () => {
  const pos = build({ "7,0": W_MAN, "6,1": B_MAN, "5,2": B_MAN });
  assert.equal(pos.legalMoves().length, 0);
  assert.ok(pos.isTerminal());
  assert.equal(pos.result(), -WHITE);
});
