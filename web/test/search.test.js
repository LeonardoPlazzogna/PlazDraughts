/**
 * The search, exercised without a network.
 *
 * With a uniform evaluator the priors carry no information, so whatever the
 * search finds it finds by looking: these tests fail if the backup sign is
 * wrong, if the virtual loss is not removed, or if terminal positions are
 * scored from the wrong side -- the three mistakes that produce a search which
 * runs, returns a move, and plays badly.
 *
 * Run: node --test web/test/search.test.js
 */
import test from "node:test";
import assert from "node:assert/strict";

import { WHITE, BLACK, W_MAN, W_KING, B_MAN, sq, N_SQUARES } from "../engine/board.js";
import { Position } from "../engine/game.js";
import { UniformEvaluator } from "../engine/evaluators.js";
import { MCTS, mulberry32 } from "../engine/mcts.js";

const board = (pieces, turn = WHITE, noProgress = 0) => {
  const b = new Int8Array(N_SQUARES);
  for (const [key, piece] of Object.entries(pieces)) {
    const [r, c] = key.split(",").map(Number);
    b[sq(r, c)] = piece;
  }
  return new Position(b, turn, noProgress);
};

test("the search finds the move that ends the game at once", async () => {
  // Black has a single man on (0,1). It can step to (1,0) -- taken -- or to
  // (1,2), and it can only capture by jumping a piece on (1,2) and landing on
  // (2,3). So White plays (2,1)->(1,2): the step is blocked, and the jump has
  // nowhere to land because (2,3) is occupied. Black cannot move and has lost.
  // Every other white move leaves the game going, so the search has to see it.
  const pos = board({
    "0,1": B_MAN,
    "1,0": W_MAN,
    "2,1": W_MAN,
    "2,3": W_MAN,          // blocks the landing square of the only possible jump
    "7,0": W_MAN,          // a spare piece, so White has other moves to choose from
  });
  const winning = pos.legalMoves().filter((m) => m.frm === sq(2, 1) && m.to === sq(1, 2));
  assert.equal(winning.length, 1, "the winning move must exist in this position");

  const mcts = new MCTS(new UniformEvaluator(), { nSims: 240, rand: mulberry32(7) });
  const ranked = await mcts.run(pos);
  assert.equal(ranked[0].move.to, sq(1, 2));
  assert.equal(ranked[0].move.frm, sq(2, 1));
  // Q lives at the CHILD, so it is the value for the side to move THERE:
  // -1 means the opponent is lost, which is what winning looks like from here.
  assert.ok(ranked[0].Q < -0.9, `a won position must score -1 at the child, got ${ranked[0].Q}`);
});

test("the search avoids the move that loses at once", async () => {
  // Mirror image: it is Black to move, and one of its moves lets White finish.
  // What matters here is the SIGN of the backup -- with it flipped, the search
  // would walk straight into the loss.
  const pos = board({
    "0,1": B_MAN, "0,5": B_MAN,
    "1,0": W_MAN,
    "2,1": W_MAN,
  }, BLACK);
  const mcts = new MCTS(new UniformEvaluator(), { nSims: 240, rand: mulberry32(11) });
  const ranked = await mcts.run(pos);
  // Moving the man on (0,1) to (1,2) hands White a capture and the game.
  const suicide = ranked.find((r) => r.move.frm === sq(0, 1) && r.move.to === sq(1, 2));
  if (suicide) {
    assert.ok(ranked[0].N > suicide.N,
      "the losing move must not be the most visited one");
  }
});

test("a forced capture leaves the search no choice", async () => {
  const pos = board({ "5,2": W_MAN, "4,3": B_MAN, "7,0": W_MAN });
  const mcts = new MCTS(new UniformEvaluator(), { nSims: 32, rand: mulberry32(3) });
  const ranked = await mcts.run(pos);
  assert.equal(ranked.length, 1, "captures are mandatory: only one legal move");
  assert.deepEqual(ranked[0].move.captured, [sq(4, 3)]);
});

test("visits add up to the simulations, and the seed makes the run repeatable", async () => {
  const pos = new Position();
  const run = async () => {
    const mcts = new MCTS(new UniformEvaluator(), { nSims: 128, batchSize: 8, rand: mulberry32(42) });
    return mcts.run(pos, { addNoise: true });
  };
  const a = await run();
  const b = await run();
  const total = a.reduce((s, r) => s + r.N, 0);
  assert.equal(total, 128, `root visits ${total} should equal the simulations`);
  assert.deepEqual(a.map((r) => [r.move.frm, r.move.to, r.N]),
                   b.map((r) => [r.move.frm, r.move.to, r.N]),
                   "same seed, same search");
});

test("a terminal root returns nothing to play", async () => {
  const pos = board({ "7,0": W_MAN, "6,1": B_MAN, "5,2": B_MAN });
  assert.ok(pos.isTerminal());
  const mcts = new MCTS(new UniformEvaluator(), { nSims: 16, rand: mulberry32(1) });
  assert.deepEqual(await mcts.run(pos), []);
});
