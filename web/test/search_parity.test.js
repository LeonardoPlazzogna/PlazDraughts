/**
 * The browser search against the Python search, move for move.
 *
 * With a uniform evaluator and no Dirichlet noise nothing in the search is
 * random: the tree is a deterministic function of the rules, the PUCT formula,
 * the leaf batching and the virtual loss. So the visit counts of the two
 * implementations must agree EXACTLY, and a single mismatch points at one of
 * those four -- the parts a port gets subtly wrong while still looking fine.
 *
 * The expected numbers come from mcts.py through
 * web/test/fixtures/make_fixtures.py; regenerate them there if the search ever
 * changes on purpose.
 *
 * Run: node --test web/test/search_parity.test.js
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { Position } from "../engine/game.js";
import { UniformEvaluator } from "../engine/evaluators.js";
import { MCTS } from "../engine/mcts.js";

const here = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(here, "fixtures", "search.json"), "utf8"));

test("the search reproduces the Python visit counts exactly", async () => {
  assert.ok(cases.length >= 4, "the fixture must cover several positions");
  for (const c of cases) {
    const pos = new Position(Int8Array.from(c.board), c.turn, c.no_progress);
    const mcts = new MCTS(new UniformEvaluator(), { nSims: c.sims, batchSize: c.batch_size });
    const ranked = await mcts.run(pos, { addNoise: false });

    const got = ranked
      .map((r) => ({ frm: r.move.frm, to: r.move.to, N: r.N, Q: r.Q }))
      .sort((a, b) => b.N - a.N || a.frm - b.frm || a.to - b.to);

    assert.equal(got.length, c.children.length, `${c.label}: number of root moves`);
    for (let i = 0; i < got.length; i++) {
      const want = c.children[i];
      assert.equal(got[i].frm, want.frm, `${c.label}: move ${i} frm`);
      assert.equal(got[i].to, want.to, `${c.label}: move ${i} to`);
      assert.equal(got[i].N, want.N, `${c.label}: ${want.frm}->${want.to} visits`);
      assert.ok(Math.abs(got[i].Q - want.Q) < 1e-9,
        `${c.label}: ${want.frm}->${want.to} value ${got[i].Q} != ${want.Q}`);
    }
  }
});

test("the visits of every case add up to the simulations asked for", async () => {
  for (const c of cases) {
    const pos = new Position(Int8Array.from(c.board), c.turn, c.no_progress);
    const ranked = await new MCTS(new UniformEvaluator(),
      { nSims: c.sims, batchSize: c.batch_size }).run(pos, { addNoise: false });
    const total = ranked.reduce((s, r) => s + r.N, 0);
    assert.equal(total, c.sims, `${c.label}: ${total} visits for ${c.sims} simulations`);
  }
});
