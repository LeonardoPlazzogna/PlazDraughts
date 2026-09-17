/**
 * AlphaZero MCTS for the browser: a FLAT tree with cached positions and leaf
 * batching. Port of mcts.py, including the two decisions that make it fast.
 *
 * NO NODE OBJECTS. The tree is a set of parallel arrays indexed by node id
 * (P, N, W, expanded, pos, term, children), and every node caches its own
 * position. The descent therefore never replays a move: each `play` happens
 * once per edge instead of once per simulation, and a `play` copies the board.
 *
 * LEAF BATCHING with a virtual loss: a wave of `batchSize` simulations collects
 * that many leaves, and the network sees them in ONE call. Do not expect much
 * from it here: measured in the browser, a batch of eight saved under a tenth
 * of the cost per position, because the forward pass dominates and the call
 * itself is cheap. It is kept because it costs nothing and because the search
 * is then the same shape as the Python one, which is what the parity test
 * compares.
 *
 * run() returns the root's children as a list of {move, N, Q}: the visit counts
 * are what the caller should play by, not the raw priors.
 */

/** Deterministic, seedable generator: the same game can be replayed exactly. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Gamma(alpha, 1) by Marsaglia-Tsang, the piece a Dirichlet draw is built from. */
function gamma(rand, alpha) {
  if (alpha < 1) {
    // Boost: Gamma(a) = Gamma(a+1) * U^(1/a), the standard way to keep alpha < 1
    // from degenerating. The default (1.5) never comes down this branch; a
    // caller asking for sharper noise would.
    return gamma(rand, alpha + 1) * Math.pow(rand() || Number.MIN_VALUE, 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      // Box-Muller for a standard normal.
      const u1 = rand() || Number.MIN_VALUE, u2 = rand();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rand() || Number.MIN_VALUE;
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function dirichlet(rand, alpha, n) {
  const g = [];
  let s = 0;
  for (let i = 0; i < n; i++) { const x = gamma(rand, alpha); g.push(x); s += x; }
  return g.map((x) => x / s);
}

export class MCTS {
  constructor(evaluator, {
    nSims = 200, cPuct = 1.5, dirichletAlpha = 1.5, dirichletEps = 0.25,
    batchSize = 8, virtualLoss = 1.0, rand = Math.random,
  } = {}) {
    this.evaluator = evaluator;
    this.nSims = nSims;
    this.cPuct = cPuct;
    this.dirichletAlpha = dirichletAlpha;
    this.dirichletEps = dirichletEps;
    this.batchSize = Math.max(1, batchSize);
    this.vloss = virtualLoss;
    this.rand = rand;
  }

  /**
   * Searches from `rootPos` and returns [{move, N, Q}], most visited first.
   * `addNoise` mixes Dirichlet noise into the root priors: it belongs to
   * self-play, not to a game against a person, so it defaults to off here.
   */
  async run(rootPos, { addNoise = false } = {}) {
    this.P = [1.0]; this.N = [0.0]; this.W = [0.0];
    this.expanded = [false];
    this.posOf = [rootPos];
    this.term = [null];
    this.cids = [null];
    this.cmoves = [null];
    // key(position) -> {priors, value}. The key is the board plus the side to
    // move and ignores the no-progress counter, exactly as the other two
    // implementations do, so a transposition reuses the first evaluation.
    this.evalCache = new Map();

    const [{ priors }] = await this.evaluator.evaluateBatch([rootPos]);
    this.expand(0, priors);
    if (!this.cids[0] || this.cids[0].length === 0) return [];
    if (addNoise) this.addDirichlet(0);

    let sims = 0;
    while (sims < this.nSims) {
      const b = Math.min(this.batchSize, this.nSims - sims);
      const pending = [];
      for (let k = 0; k < b; k++) {
        const { leaf, path } = this.selectToLeaf();
        if (this.term[leaf] !== null) this.backup(path, this.term[leaf]);
        else pending.push({ leaf, path });
      }
      sims += b;
      if (pending.length === 0) continue;

      const miss = [];
      for (const { leaf, path } of pending) {
        const cached = this.evalCache.get(this.posOf[leaf].key());
        if (cached) {
          if (!this.expanded[leaf]) this.expand(leaf, cached.priors);
          this.backup(path, cached.value);
        } else {
          miss.push({ leaf, path });
        }
      }
      if (miss.length > 0) {
        const res = await this.evaluator.evaluateBatch(miss.map((m) => this.posOf[m.leaf]));
        for (let i = 0; i < miss.length; i++) {
          const { leaf, path } = miss[i];
          const { priors, value } = res[i];
          this.evalCache.set(this.posOf[leaf].key(), { priors, value });
          if (!this.expanded[leaf]) this.expand(leaf, priors);
          this.backup(path, value);
        }
      }
    }

    const out = this.cids[0].map((cid, k) => ({
      move: this.cmoves[0][k],
      N: this.N[cid],
      Q: this.N[cid] > 0 ? this.W[cid] / this.N[cid] : 0.0,
    }));
    out.sort((a, b) => b.N - a.N);
    return out;
  }

  // --- internals -------------------------------------------------------
  newNode(prior) {
    this.P.push(prior); this.N.push(0.0); this.W.push(0.0);
    this.expanded.push(false); this.posOf.push(null);
    this.term.push(null); this.cids.push(null); this.cmoves.push(null);
    return this.P.length - 1;
  }

  expand(i, priors) {
    const cids = [], cmoves = [];
    for (const { move, prior } of priors) {
      cids.push(this.newNode(prior));
      cmoves.push(move);
    }
    this.cids[i] = cids;
    this.cmoves[i] = cmoves;
    this.expanded[i] = true;
  }

  addDirichlet(i) {
    const cids = this.cids[i];
    const noise = dirichlet(this.rand, this.dirichletAlpha, cids.length);
    const eps = this.dirichletEps;
    cids.forEach((cid, k) => { this.P[cid] = (1 - eps) * this.P[cid] + eps * noise[k]; });
  }

  selectToLeaf() {
    let i = 0;
    const path = [0];
    const { N, W, P, cPuct } = this;
    while (this.expanded[i]) {
      const cids = this.cids[i];
      const ni = N[i];
      const sqrtN = ni > 1.0 ? Math.sqrt(ni) : 1.0;
      let best = -1e30, bk = 0;
      for (let k = 0; k < cids.length; k++) {
        const cid = cids[k];
        const n = N[cid];
        const q = n > 0 ? -(W[cid] / n) : 0.0;
        const s = q + cPuct * P[cid] * sqrtN / (1.0 + n);
        if (s > best) { best = s; bk = k; }
      }
      const cid = cids[bk];
      if (this.posOf[cid] === null) {            // realize the edge, once
        const childPos = this.posOf[i].play(this.cmoves[i][bk]);
        this.posOf[cid] = childPos;
        if (childPos.isTerminal()) this.term[cid] = childPos.terminalValue(childPos.turn);
      }
      N[cid] += 1.0; W[cid] += this.vloss;       // virtual loss
      path.push(cid);
      i = cid;
    }
    return { leaf: i, path };
  }

  backup(path, value) {
    const { N, W, vloss } = this;
    for (let k = 1; k < path.length; k++) { N[path[k]] -= 1.0; W[path[k]] -= vloss; }
    let v = value;
    for (let k = path.length - 1; k >= 0; k--) {
      N[path[k]] += 1.0; W[path[k]] += v; v = -v;
    }
  }
}
