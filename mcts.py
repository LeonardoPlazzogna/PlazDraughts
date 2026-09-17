"""
AlphaZero MCTS for Italian draughts: a FLAT tree with cached positions, plus
leaf batching.

No `Node` object per node: the tree is a set of PARALLEL LISTS indexed by node
id (P, N, W, expanded, pos, term, children), and EVERY node CACHES its own
position. The descent therefore does NOT re-run `pos.play` at every simulation
(that was the dominant cost: copying the board): each `play` happens once per
edge, on the first visit. It also avoids allocating an object and a children
dict per node.

Leaf batching with virtual loss: one network evaluation per "wave" of
`batch_size` simulations.

run() returns an object whose `.children` maps each root Move to its visit
statistics (.N, .Q), the interface used by self-play and the tools.
"""
from __future__ import annotations
import math
import numpy as np


class _Stat:
    __slots__ = ("N", "Q")

    def __init__(self, N, Q):
        self.N = N
        self.Q = Q


class _Root:
    __slots__ = ("children",)

    def __init__(self, children):
        self.children = children


class MCTS:
    def __init__(self, evaluator, n_sims: int = 200, c_puct: float = 1.5,
                 dirichlet_alpha: float = 1.5, dirichlet_eps: float = 0.25,
                 batch_size: int = 8, virtual_loss: float = 1.0,
                 rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.batch_size = max(1, batch_size)
        self.vloss = virtual_loss
        self.rng = rng or np.random.default_rng()

    def run(self, root_pos, add_noise: bool = True) -> _Root:
        # parallel lists indexed by node id (0 = root)
        self.P = [1.0]
        self.N = [0.0]
        self.W = [0.0]
        self.expanded = [False]
        self.pos = [root_pos]      # CACHED position (None until realized)
        self.term = [None]         # terminal value (float), or None if not terminal
        self.cids = [None]         # child ids
        self.cmoves = [None]       # children's Moves (parallel to cids)
        # key(position) -> (priors, value), for transpositions. The key is the
        # board plus the side to move and ignores the no-progress counter: a
        # transposition reached with a different counter reuses the first
        # evaluation (the C++ engine does the same).
        self._eval_cache = {}

        priors, _ = self.evaluator.evaluate_batch([root_pos])[0]
        self._expand(0, priors)
        if not self.cids[0]:
            return _Root({})
        if add_noise:
            self._add_dirichlet(0)

        sims = 0
        while sims < self.n_sims:
            b = min(self.batch_size, self.n_sims - sims)
            pending = []
            for _ in range(b):
                leaf, path = self._select_to_leaf()
                if self.term[leaf] is not None:
                    self._backup(path, self.term[leaf])
                else:
                    pending.append((leaf, path))
            sims += b
            if pending:
                # skip the forward pass for positions already evaluated (transpositions)
                miss = []
                for leaf, path in pending:
                    cached = self._eval_cache.get(self.pos[leaf].key())
                    if cached is not None:
                        priors, value = cached
                        if not self.expanded[leaf]:
                            self._expand(leaf, priors)
                        self._backup(path, value)
                    else:
                        miss.append((leaf, path))
                if miss:
                    res = self.evaluator.evaluate_batch([self.pos[l] for (l, _) in miss])
                    for (leaf, path), (priors, value) in zip(miss, res):
                        self._eval_cache[self.pos[leaf].key()] = (priors, value)
                        if not self.expanded[leaf]:
                            self._expand(leaf, priors)
                        self._backup(path, value)

        children = {mv: _Stat(self.N[cid],
                              self.W[cid] / self.N[cid] if self.N[cid] > 0 else 0.0)
                    for cid, mv in zip(self.cids[0], self.cmoves[0])}
        return _Root(children)

    # --- internals --------------------------------------------------------
    def _new_node(self, prior: float) -> int:
        self.P.append(prior); self.N.append(0.0); self.W.append(0.0)
        self.expanded.append(False); self.pos.append(None)
        self.term.append(None); self.cids.append(None); self.cmoves.append(None)
        return len(self.P) - 1

    def _expand(self, i: int, priors: dict) -> None:
        cids, cmoves = [], []
        for mv, p in priors.items():
            cids.append(self._new_node(p))
            cmoves.append(mv)
        self.cids[i] = cids
        self.cmoves[i] = cmoves
        self.expanded[i] = True

    def _add_dirichlet(self, i: int) -> None:
        cids = self.cids[i]
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(cids))
        eps = self.dirichlet_eps
        for cid, nz in zip(cids, noise):
            self.P[cid] = (1.0 - eps) * self.P[cid] + eps * float(nz)

    def _select_to_leaf(self):
        i = 0
        path = [0]
        c_puct = self.c_puct
        N, W, P = self.N, self.W, self.P
        while self.expanded[i]:
            cids = self.cids[i]
            ni = N[i]
            sqrt_n = math.sqrt(ni) if ni > 1.0 else 1.0
            best, bk = -1e30, 0
            for k in range(len(cids)):
                cid = cids[k]
                n = N[cid]
                q = -(W[cid] / n) if n > 0 else 0.0
                s = q + c_puct * P[cid] * sqrt_n / (1.0 + n)
                if s > best:
                    best, bk = s, k
            cid = cids[bk]
            if self.pos[cid] is None:                       # realize (one play per edge)
                child_pos = self.pos[i].play(self.cmoves[i][bk])
                self.pos[cid] = child_pos
                if child_pos.is_terminal():
                    self.term[cid] = child_pos.terminal_value(child_pos.turn)
            N[cid] += 1.0; W[cid] += self.vloss             # virtual loss
            path.append(cid)
            i = cid
        return i, path

    def _backup(self, path, value: float) -> None:
        vloss = self.vloss
        N, W = self.N, self.W
        for cid in path[1:]:                                # remove the virtual loss
            N[cid] -= 1.0; W[cid] -= vloss
        v = value
        for cid in reversed(path):
            N[cid] += 1.0; W[cid] += v; v = -v
