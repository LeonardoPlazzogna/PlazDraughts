/**
 * Running a trained network in the page, through ONNX Runtime Web.
 *
 * This is the only file that knows the network is ONNX and that it runs in
 * WebAssembly. The search asks an evaluator for priors and a value; what
 * produces them stays here, so the rest of the engine is testable without a
 * model and works unchanged if the runtime is ever swapped.
 *
 * The runtime is the copy in web/vendor/, not a CDN, and the model files are
 * the ones tools/export_onnx.py wrote after checking them against PyTorch.
 */
import { encode, IN_PLANES } from "./encoder.js";
import { NetEvaluator } from "./evaluators.js";

const PLANE_VALUES = IN_PLANES * 64;

let ortPromise = null;

/**
 * The runtime, loaded once.
 *
 * The ES-module build is used, not the classic script: a module can be imported
 * from a page AND from a module worker, and the search has to run in a worker
 * or the board freezes while the engine thinks.
 *
 * Paths are resolved against THIS file, not against the page. `wasmPaths` in
 * particular must end up absolute: a relative one is resolved against the
 * runtime's own script, so "./vendor/" becomes "vendor/vendor/" and the only
 * thing the runtime then says is "no available backend found".
 */
export function loadRuntime(vendorPath = "../vendor/") {
  if (ortPromise) return ortPromise;
  const base = new URL(vendorPath, import.meta.url);
  ortPromise = import(new URL("ort.wasm.min.mjs", base).href).then((mod) => {
    const ort = mod.default ?? mod;
    ort.env.wasm.wasmPaths = base.href;
    // One thread: SharedArrayBuffer needs cross-origin isolation, and GitHub
    // Pages cannot send those headers. Asking for more only earns a warning.
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";
    return ort;
  });
  return ortPromise;
}

/** Opens a model. `url` is a path like "models/runB-c200.onnx". */
export async function createSession(url, { vendorPath = "../vendor/" } = {}) {
  const ort = await loadRuntime(vendorPath);
  const session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return { ort, session, url };
}

/** The planes of a batch of positions, laid out as the model expects them. */
export function encodeBatch(positions) {
  const planes = new Float32Array(positions.length * PLANE_VALUES);
  positions.forEach((pos, i) => planes.set(encode(pos), i * PLANE_VALUES));
  return planes;
}

/**
 * A runner for NetEvaluator: planes in, policy and value out.
 *
 * The first call pays for warming up the runtime, so callers that care about
 * the first move should run one throwaway batch while the page is still
 * loading.
 */
export function makeRunner({ ort, session }) {
  return async function run(planes, batch) {
    const input = new ort.Tensor("float32", planes, [batch, IN_PLANES, 8, 8]);
    const out = await session.run({ planes: input });
    return { policy: out.policy.data, value: out.value.data };
  };
}

/** Everything at once: an evaluator the search can use. */
export async function createNetEvaluator(url, opts = {}) {
  const ctx = await createSession(url, opts);
  const evaluator = new NetEvaluator(makeRunner(ctx), encodeBatch);
  return { evaluator, ...ctx };
}
