/**
 * Evaluators: what the search asks for a position.
 *
 * The contract is one method, `evaluateBatch(positions)`, returning for each
 * position `{priors, value}` where `priors` is a list of {move, prior} over the
 * LEGAL moves and `value` is in [-1, 1] from the point of view of the side to
 * move. Keeping it to one batched method is what lets the search collect a wave
 * of leaves and pay for a single call into the network.
 */
import { priorsFromLogits } from "./encoder.js";

/**
 * No network: every legal move equally likely, every position even.
 *
 * It is not a toy. It gives a playable opponent before any model is downloaded,
 * it is what the tests use to exercise the search on its own, and it is what the
 * worker falls back to when a model will not load -- the game goes on, and the
 * page says what it is really playing against.
 */
export class UniformEvaluator {
  async evaluateBatch(positions) {
    return positions.map((pos) => {
      const moves = pos.legalMoves();
      const p = moves.length > 0 ? 1 / moves.length : 0;
      return { priors: moves.map((move) => ({ move, prior: p })), value: 0.0 };
    });
  }
}

/**
 * A network evaluator: whatever runs the ONNX model, wrapped so the search does
 * not know about it. `runner` takes a Float32Array of B*7*8*8 planes and the
 * batch size, and resolves to {policy, value} as flat Float32Arrays.
 */
export class NetEvaluator {
  constructor(runner, encodeFn) {
    this.runner = runner;
    this.encode = encodeFn;
  }

  async evaluateBatch(positions) {
    const planes = this.encode(positions);
    const { policy, value } = await this.runner(planes, positions.length);
    const policySize = policy.length / positions.length;
    return positions.map((pos, i) => ({
      priors: priorsFromLogits(pos, policy.subarray(i * policySize, (i + 1) * policySize)),
      value: value[i],
    }));
  }
}
