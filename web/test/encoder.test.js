/**
 * The browser encoder against the checksums the C++ engine is held to.
 *
 * The network is the same file in both places, so the only way the browser can
 * feed it something different is the encoding: a plane in the wrong order, a
 * missing rotation for Black, a counter normalized differently. None of that
 * crashes -- it just makes the engine play worse, quietly.
 *
 * The oracle is engine_c/encoder_parity.cpp: the same cases, the same checksum
 * (sum and index-weighted sum of the 448 values), the same expected numbers,
 * generated once with dama/encoder.py.
 *
 * Run: node --test web/test/encoder.test.js
 */
import test from "node:test";
import assert from "node:assert/strict";

import { WHITE, BLACK, W_MAN, W_KING, B_MAN, B_KING, sq, N_SQUARES } from "../engine/board.js";
import { Position } from "../engine/game.js";
import { encode, legalActions, indexToMove, priorsFromLogits, POLICY_SIZE } from "../engine/encoder.js";

/** sum and index-weighted sum of the flattened planes, as the C++ computes them. */
function checksum(planes) {
  let sum = 0, wsum = 0;
  for (let i = 0; i < planes.length; i++) { sum += planes[i]; wsum += planes[i] * i; }
  return { sum, wsum };
}

function syntheticBoard() {
  const b = new Int8Array(N_SQUARES);
  b[sq(4, 3)] = W_MAN;     // can capture the black man below, if White is to move
  b[sq(3, 2)] = B_MAN;
  b[sq(5, 4)] = W_KING;
  b[sq(2, 3)] = B_KING;
  return b;
}

test("encoder checksums match the Python reference", () => {
  const cases = [
    ["start, White, np=0", new Position(null, WHITE, 0), 56.0, 11492.0],
    ["start, Black, np=0", new Position(null, BLACK, 0), 56.0, 11492.0],
    ["synthetic, White, np=20", new Position(syntheticBoard(), WHITE, 20), 116.0, 38852.0],
    ["synthetic, Black, np=20", new Position(syntheticBoard(), BLACK, 20), 52.0, 16360.0],
  ];
  for (const [label, pos, expSum, expWsum] of cases) {
    const { sum, wsum } = checksum(encode(pos));
    assert.ok(Math.abs(sum - expSum) < 1e-4, `${label}: sum ${sum} != ${expSum}`);
    assert.ok(Math.abs(wsum - expWsum) < 1e-4, `${label}: wsum ${wsum} != ${expWsum}`);
  }
});

test("encoder checksums along a deterministic playout", () => {
  // The move is chosen by value -- lowest frm, then lowest to -- so that every
  // implementation walks the same game even if its generator returns the moves
  // in another order. Checking only hand-built positions would leave the chain
  // generate -> apply -> encode untested on positions the game really reaches.
  const expected = new Map([
    [5, [55.799999, 11659.400391]],
    [10, [53.799999, 11616.400391]],
    [15, [118.399994, 34448.199219]],
    [20, [114.0, 33712.0]],
  ]);
  let pos = new Position();
  let checked = 0;
  for (let ply = 0; ply < 20; ply++) {
    if (pos.isTerminal()) break;
    const moves = pos.legalMoves();
    let best = moves[0];
    for (const m of moves) {
      if (m.frm < best.frm || (m.frm === best.frm && m.to < best.to)) best = m;
    }
    pos = pos.play(best);
    const want = expected.get(ply + 1);
    if (!want) continue;
    const { sum, wsum } = checksum(encode(pos));
    assert.ok(Math.abs(sum - want[0]) < 1e-3, `ply ${ply + 1}: sum ${sum} != ${want[0]}`);
    assert.ok(Math.abs(wsum - want[1]) < 1e-2, `ply ${ply + 1}: wsum ${wsum} != ${want[1]}`);
    checked++;
  }
  assert.equal(checked, 4, "all four playout checkpoints must be reached");
});

test("policy indices stay inside the compact action space and map back", () => {
  let pos = new Position();
  for (let ply = 0; ply < 40 && !pos.isTerminal(); ply++) {
    const { moves, indices } = legalActions(pos);
    for (let i = 0; i < moves.length; i++) {
      assert.ok(indices[i] >= 0 && indices[i] < POLICY_SIZE, "index out of range");
      const back = indexToMove(pos, indices[i]);
      assert.ok(back, "an index of a legal move must map back to a move");
      // Several chains can share a first step: what must match is that step.
      assert.equal(back.frm, moves[i].frm);
      assert.equal(back.path[1], moves[i].path[1]);
    }
    pos = pos.play(moves[ply % moves.length]);
  }
});

test("priors cover the legal moves and sum to one", () => {
  // Collisions included: two chains sharing a first jump share a logit, and the
  // probability must be split between them, not counted twice.
  const board = new Int8Array(N_SQUARES);
  board[sq(5, 4)] = W_KING;
  board[sq(4, 3)] = B_MAN; board[sq(2, 3)] = B_MAN;
  board[sq(4, 5)] = B_MAN; board[sq(2, 5)] = B_MAN;
  for (const pos of [new Position(), new Position(board, WHITE, 0)]) {
    const logits = new Float32Array(POLICY_SIZE);
    for (let i = 0; i < POLICY_SIZE; i++) logits[i] = Math.sin(i) * 2;
    const priors = priorsFromLogits(pos, logits);
    assert.equal(priors.length, pos.legalMoves().length);
    const total = priors.reduce((s, p) => s + p.prior, 0);
    assert.ok(Math.abs(total - 1) < 1e-9, `priors sum to ${total}`);
    assert.ok(priors.every((p) => p.prior > 0));
  }
});
