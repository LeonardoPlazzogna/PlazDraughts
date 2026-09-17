/**
 * The engine, in a worker.
 *
 * The search is hundreds of network calls and it runs for a second or more.
 * On the page's own thread that means a board that does not repaint, a button
 * that does not react and, on a phone, a browser that offers to kill the tab.
 * Here it runs beside the page instead, which stays responsive and can show
 * what is going on.
 *
 * The protocol is three messages:
 *   {type:"load",  model}  -> {type:"ready"}          the network, or none
 *   {type:"think", ...}    -> {type:"move", ...}      one search
 *   anything failing       -> {type:"error", message}
 *
 * Positions cross as plain arrays, because a Position carries methods and a
 * memoized move list that cannot be cloned.
 */
import { Position } from "../engine/game.js";
import { MCTS } from "../engine/mcts.js";
import { UniformEvaluator } from "../engine/evaluators.js";
import { createNetEvaluator } from "../engine/net.js";

let evaluator = new UniformEvaluator();

const asPosition = (d) => new Position(Int8Array.from(d.board), d.turn, d.noProgress);

/** A move, reduced to what the page needs to draw and to record it. */
const plainMove = (m) => ({
  frm: m.frm, to: m.to, path: Array.from(m.path),
  captured: Array.from(m.captured), promotes: m.promotes, byKing: m.byKing,
  isCapture: m.isCapture,
});

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") {
      if (!msg.model) {
        evaluator = new UniformEvaluator();
        self.postMessage({ type: "ready", model: null });
        return;
      }
      try {
        const { evaluator: net } = await createNetEvaluator(msg.model);
        evaluator = net;
        // One throwaway evaluation: the first call through the runtime pays for
        // compiling the model, and that cost should land here, while the page
        // says "loading", not on the engine's first move.
        await evaluator.evaluateBatch([new Position()]);
        self.postMessage({ type: "ready", model: msg.model });
      } catch (err) {
        // A network that will not load must not take the game down with it:
        // the search works without one, so it falls back and the page is told
        // what it is really playing against.
        evaluator = new UniformEvaluator();
        self.postMessage({
          type: "ready", model: null,
          failed: String(err && err.message ? err.message : err),
        });
      }
      return;
    }

    if (msg.type === "think") {
      const pos = asPosition(msg);
      const t0 = performance.now();
      const ranked = await new MCTS(evaluator, {
        nSims: msg.sims, batchSize: 8, cPuct: 1.5,
      }).run(pos, { addNoise: false });
      if (ranked.length === 0) {
        self.postMessage({ type: "move", move: null, seconds: 0 });
        return;
      }
      // The engine plays the most visited move -- the visits, not the priors,
      // are what the search actually concluded.
      const best = ranked[0];
      self.postMessage({
        type: "move",
        move: plainMove(best.move),
        // Q is the value for the side to move AFTER the move, so the score for
        // the side that just played is its opposite.
        score: -best.Q,
        visits: best.N,
        considered: ranked.length,
        seconds: (performance.now() - t0) / 1000,
      });
      return;
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  }
};
