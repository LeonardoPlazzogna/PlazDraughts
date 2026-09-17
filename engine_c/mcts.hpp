// AlphaZero MCTS for Italian draughts. Faithful port of mcts.py: a FLAT tree
// (parallel vectors indexed by node id, not one object per node), every node
// caches its realized position, leaf batching + virtual loss, and a cache of the
// evaluations for transpositions within a single search. PUCT with no shuffling
// of the children, no progressive widening, no subtree reuse between moves, no
// early stopping: deliberate choices, consistent with a branching factor much
// smaller than in chess (see chapter 2 of the thesis).
//
// The tree is written against an abstract IEvaluator interface, so it is fully
// testable (see UniformEvaluator below and mcts_selfplay_test.cpp) WITHOUT any
// LibTorch dependency. The network-backed evaluator (HubEvaluator, in
// inference_hub.hpp) is a separate implementation of the same interface.
#pragma once
#include "position.hpp"
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include <algorithm>

namespace dama {

// Result of an evaluation: priors over the legal moves (same order as
// Position::legal_moves(), summing to 1) + a value in [-1,1] from the point of
// view of the side to move.
struct EvalResult {
    std::vector<float> priors;   // priors[i] corresponds to legal_moves()[i]
    float value;
};

class IEvaluator {
public:
    virtual ~IEvaluator() = default;
    virtual std::vector<EvalResult> evaluate_batch(const std::vector<Position>& positions) = 0;
};

// Trivial evaluator: uniform priors over the legal moves, value 0. Used to test
// the tree (selection, expansion, backup, terminal handling) without a neural
// network -- the same role as UniformEvaluator in evaluators.py.
class UniformEvaluator : public IEvaluator {
public:
    std::vector<EvalResult> evaluate_batch(const std::vector<Position>& positions) override {
        std::vector<EvalResult> out;
        out.reserve(positions.size());
        for (const auto& pos : positions) {
            const auto& moves = pos.legal_moves();
            EvalResult r;
            r.value = 0.0f;
            if (!moves.empty())
                r.priors.assign(moves.size(), 1.0f / static_cast<float>(moves.size()));
            out.push_back(std::move(r));
        }
        return out;
    }
};

struct MCTSConfig {
    int n_sims = 200;
    float c_puct = 1.5f;
    float dirichlet_alpha = 1.5f;
    float dirichlet_eps = 0.25f;
    int batch_size = 8;
    float virtual_loss = 1.0f;
};

struct ChildStat { double N = 0.0; double Q = 0.0; };

class MCTS {
public:
    MCTS(IEvaluator& evaluator, MCTSConfig cfg, std::mt19937& rng)
        : ev_(evaluator), cfg_(cfg), rng_(rng) {}

    // Full search from the given position. Returns the visit statistics of the
    // root's children (move -> N, Q), like root.children in mcts.py. If the
    // root has no legal moves, returns an empty vector.
    std::vector<std::pair<Move, ChildStat>> run(const Position& root_pos, bool add_noise) {
        P_.assign(1, 1.0f); N_.assign(1, 0.0); W_.assign(1, 0.0);
        expanded_.assign(1, false);
        pos_.assign(1, root_pos);
        term_.assign(1, std::nullopt);
        cids_.assign(1, {}); cmoves_.assign(1, {});
        eval_cache_.clear();

        auto root_eval = ev_.evaluate_batch({root_pos});
        expand(0, root_pos.legal_moves(), root_eval[0].priors);
        if (cids_[0].empty()) return {};
        if (add_noise) add_dirichlet(0);

        int sims = 0;
        while (sims < cfg_.n_sims) {
            int b = std::min(cfg_.batch_size, cfg_.n_sims - sims);
            std::vector<std::pair<int, std::vector<int>>> pending;
            for (int i = 0; i < b; i++) {
                auto [leaf, path] = select_to_leaf();
                if (term_[leaf].has_value())
                    backup(path, *term_[leaf]);
                else
                    pending.push_back({leaf, std::move(path)});
            }
            sims += b;
            if (pending.empty()) continue;

            // skip the forward pass for positions already evaluated
            // (transpositions; the key ignores the no-progress counter, as in mcts.py)
            std::vector<std::pair<int, std::vector<int>>> miss;
            for (auto& [leaf, path] : pending) {
                auto it = eval_cache_.find(key_of(*pos_[leaf]));
                if (it != eval_cache_.end()) {
                    if (!expanded_[leaf])
                        expand(leaf, pos_[leaf]->legal_moves(), it->second.priors);
                    backup(path, it->second.value);
                } else {
                    miss.push_back({leaf, std::move(path)});
                }
            }
            if (miss.empty()) continue;

            std::vector<Position> batch_pos;
            batch_pos.reserve(miss.size());
            for (auto& [leaf, path] : miss) batch_pos.push_back(*pos_[leaf]);
            auto results = ev_.evaluate_batch(batch_pos);

            for (size_t i = 0; i < miss.size(); i++) {
                int leaf = miss[i].first;
                auto& path = miss[i].second;
                const EvalResult& res = results[i];
                eval_cache_[key_of(*pos_[leaf])] = res;
                if (!expanded_[leaf])
                    expand(leaf, pos_[leaf]->legal_moves(), res.priors);
                backup(path, res.value);
            }
        }

        std::vector<std::pair<Move, ChildStat>> children;
        children.reserve(cids_[0].size());
        for (size_t k = 0; k < cids_[0].size(); k++) {
            int cid = cids_[0][k];
            ChildStat st;
            st.N = N_[cid];
            st.Q = (N_[cid] > 0.0) ? (W_[cid] / N_[cid]) : 0.0;
            children.push_back({cmoves_[0][k], st});
        }
        return children;
    }

private:
    IEvaluator& ev_;
    MCTSConfig cfg_;
    std::mt19937& rng_;

    std::vector<float> P_;
    std::vector<double> N_, W_;
    std::vector<bool> expanded_;
    std::vector<std::optional<Position>> pos_;
    std::vector<std::optional<float>> term_;
    std::vector<std::vector<int>> cids_;
    std::vector<std::vector<Move>> cmoves_;
    std::unordered_map<PositionKey, EvalResult, PositionKeyHash> eval_cache_;

    int new_node(float prior) {
        P_.push_back(prior); N_.push_back(0.0); W_.push_back(0.0);
        expanded_.push_back(false); pos_.push_back(std::nullopt);
        term_.push_back(std::nullopt); cids_.push_back({}); cmoves_.push_back({});
        return static_cast<int>(P_.size()) - 1;
    }

    void expand(int i, const std::vector<Move>& moves_in, const std::vector<float>& priors_in) {
        // Defensive copy FIRST, before any call to new_node(): the incoming
        // references may point inside pos_[some_id] (e.g.
        // pos_[leaf]->legal_moves()), and new_node() does push_back on
        // pos_/P_/... which can reallocate and invalidate them halfway through.
        std::vector<Move> moves = moves_in;
        std::vector<float> priors = priors_in;

        std::vector<int> cids;
        cids.reserve(moves.size());
        for (size_t k = 0; k < moves.size(); k++)
            cids.push_back(new_node(priors[k]));
        cids_[i] = std::move(cids);
        cmoves_[i] = std::move(moves);
        expanded_[i] = true;
    }

    void add_dirichlet(int i) {
        const auto& cids = cids_[i];
        int n = static_cast<int>(cids.size());
        std::gamma_distribution<double> gamma(cfg_.dirichlet_alpha, 1.0);
        std::vector<double> g(n);
        double sum = 0.0;
        for (int k = 0; k < n; k++) { g[k] = gamma(rng_); sum += g[k]; }
        float eps = cfg_.dirichlet_eps;
        for (int k = 0; k < n; k++) {
            float nz = static_cast<float>(g[k] / sum);
            P_[cids[k]] = (1.0f - eps) * P_[cids[k]] + eps * nz;
        }
    }

    std::pair<int, std::vector<int>> select_to_leaf() {
        int i = 0;
        std::vector<int> path{0};
        while (expanded_[i]) {
            const auto& cids = cids_[i];
            double ni = N_[i];
            double sqrt_n = (ni > 1.0) ? std::sqrt(ni) : 1.0;
            double best = -1e30;
            int bk = 0;
            for (size_t k = 0; k < cids.size(); k++) {
                int cid = cids[k];
                double n = N_[cid];
                double q = (n > 0.0) ? -(W_[cid] / n) : 0.0;
                double s = q + cfg_.c_puct * P_[cid] * sqrt_n / (1.0 + n);
                if (s > best) { best = s; bk = static_cast<int>(k); }
            }
            int cid = cids[bk];
            if (!pos_[cid].has_value()) {          // realize (one play per edge)
                Position child_pos = pos_[i]->play(cmoves_[i][bk]);
                bool term = child_pos.is_terminal();
                int child_turn = child_pos.turn;
                pos_[cid] = std::move(child_pos);
                if (term) term_[cid] = pos_[cid]->terminal_value(child_turn);
            }
            N_[cid] += 1.0; W_[cid] += cfg_.virtual_loss;   // virtual loss
            path.push_back(cid);
            i = cid;
        }
        return {i, path};
    }

    void backup(const std::vector<int>& path, float value) {
        for (size_t k = 1; k < path.size(); k++) {          // remove the virtual loss
            N_[path[k]] -= 1.0; W_[path[k]] -= cfg_.virtual_loss;
        }
        double v = value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            N_[*it] += 1.0; W_[*it] += v; v = -v;
        }
    }
};

}  // namespace dama
