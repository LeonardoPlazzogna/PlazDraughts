// Self-play and generation of the training samples. Faithful port of selfplay.py.
//
// Each sample is (X, pi, mask, z):
//   X    : [7,8,8] tensor of the position (canonical encoder)
//   pi   : policy target [256] = MCTS visit distribution
//   mask : [256] legal actions (for the masking in the loss)
//   z    : final outcome from the point of view of the side to move (+1/-1/0,
//          scaled by gamma^d when value_discount < 1)
//
// It also contains the MULTI-THREAD driver: N threads play games in parallel,
// each with its own MCTS tree and its own evaluator connected to the hub
// (inference_hub.hpp). This is where the C++ engine departs from the Python
// pipeline: no separate processes, no IPC, shared memory.
#pragma once
#include "position.hpp"
#include "mcts.hpp"
#include "encoder.hpp"
#include <vector>
#include <array>
#include <random>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cmath>

namespace dama {

constexpr int PLANE_FLOATS = ENC_IN_PLANES * 8 * 8;

struct Sample {
    std::array<float, PLANE_FLOATS> x{};
    std::array<float, ENC_POLICY_SIZE> pi{};
    std::array<float, ENC_POLICY_SIZE> mask{};
    float z = 0.0f;
};

struct GameResult {
    std::vector<Sample> samples;
    int z_white = 0;
    int plies = 0;
    // Diagnostics (the same fields the Python path collects, selfplay.py):
    // they judge the QUALITY of the data, not just its quantity.
    bool truncated = false;      // max_plies reached without an end
    bool no_progress = false;    // draw by the no-progress counter
    int captures = 0;
    int promotions = 0;
    double entropy_sum = 0.0;    // sum of the visit-policy entropies
    double q_spread_sum = 0.0;   // sum of the max-min spreads of Q at the root
    double value_abs_err = 0.0;  // sum of |search value - outcome|
};

struct SelfPlayConfig {
    int n_sims = 400;
    int temp_moves = 12;       // opening plies played with temperature 1
    float c_puct = 1.5f;
    int batch_size = 8;        // MCTS leaf batching
    int max_plies = 300;
    float dirichlet_alpha = 1.5f;
    float dirichlet_eps = 0.25f;
    // Shrinks the value target with the distance from the end: a position d
    // plies before the end gets z * gamma^d. 1.0 = off.
    //
    // Without a discount, in a won position every move is worth +1 and the
    // search has no way to prefer one: with four kings against one every legal
    // move is valued +1.000 and the visits spread almost evenly, so the engine
    // shuffles into a no-progress draw. With the discount, winning sooner is
    // worth more than winning later, and the tie is broken.
    float value_discount = 1.0f;
};

// Adjudicates a TRUNCATED game (max_plies reached without an end) by material
// balance (king = 3). +1 White, -1 Black, 0 even, from White's point of view,
// consistent with result().
inline int material_result(const Board& b) {
    int m = 0;
    for (int8_t p : b) {
        if (p == W_MAN) m += 1;
        else if (p == W_KING) m += 3;
        else if (p == B_MAN) m -= 1;
        else if (p == B_KING) m -= 3;
    }
    return (m > 0) ? 1 : ((m < 0) ? -1 : 0);
}

// Visit distribution over the root's children, as a [256] vector of compact
// actions (same frame as the encoder).
inline std::array<float, ENC_POLICY_SIZE> visit_policy_target(
        const Position& pos, const std::vector<std::pair<Move, ChildStat>>& children) {
    std::array<float, ENC_POLICY_SIZE> target{};
    bool flip = (pos.turn == BLACK);
    double total = 0.0;
    for (const auto& [mv, st] : children) {
        target[move_policy_index(mv, flip)] += (float)st.N;
        total += st.N;
    }
    if (total > 0.0)
        for (auto& v : target) v = (float)(v / total);
    return target;
}

inline std::array<float, ENC_POLICY_SIZE> legal_mask_of(const Position& pos) {
    std::array<float, ENC_POLICY_SIZE> mask{};
    bool flip = (pos.turn == BLACK);
    for (const auto& mv : pos.legal_moves())
        mask[move_policy_index(mv, flip)] = 1.0f;
    return mask;
}

// Move selection from the visits: temperature 1 at the start (exploration),
// then greedy (argmax). Same logic as selfplay.py:select_move.
inline Move select_move(const std::vector<std::pair<Move, ChildStat>>& children,
                        float temperature, std::mt19937& rng) {
    if (temperature <= 1e-6f) {
        auto best = std::max_element(children.begin(), children.end(),
            [](const auto& a, const auto& b) { return a.second.N < b.second.N; });
        return best->first;
    }
    std::vector<double> w;
    w.reserve(children.size());
    double sum = 0.0;
    for (const auto& [mv, st] : children) {
        double v = std::pow(st.N, 1.0 / temperature);
        w.push_back(v);
        sum += v;
    }
    if (sum <= 0.0) return children.front().first;
    std::uniform_real_distribution<double> ud(0.0, sum);
    double r = ud(rng);
    double acc = 0.0;
    for (size_t i = 0; i < children.size(); i++) {
        acc += w[i];
        if (r <= acc) return children[i].first;
    }
    return children.back().first;
}

// Plays one complete self-play game.
inline GameResult play_game(IEvaluator& ev, const SelfPlayConfig& cfg, unsigned seed) {
    std::mt19937 rng(seed);
    MCTSConfig mc;
    mc.n_sims = cfg.n_sims;
    mc.c_puct = cfg.c_puct;
    mc.batch_size = cfg.batch_size;
    mc.dirichlet_alpha = cfg.dirichlet_alpha;
    mc.dirichlet_eps = cfg.dirichlet_eps;

    Position pos;
    GameResult res;
    std::vector<int> turns;             // side to move of each sample
    std::vector<double> root_values;    // value estimated by the search

    while (!pos.is_terminal() && res.plies < cfg.max_plies) {
        MCTS mcts(ev, mc, rng);
        auto children = mcts.run(pos, /*add_noise=*/true);
        if (children.empty()) break;

        Sample s;
        Planes pl = encode(pos.board, pos.turn, pos.no_progress, NO_PROGRESS_DRAW);
        std::copy(pl.begin(), pl.end(), s.x.begin());
        s.pi = visit_policy_target(pos, children);
        s.mask = legal_mask_of(pos);
        res.samples.push_back(s);
        turns.push_back(pos.turn);

        // entropy of the visit distribution: how decisive the search is
        double h = 0.0;
        for (float p : s.pi) if (p > 0.0f) h -= (double)p * std::log((double)p);
        res.entropy_sum += h;

        // root value: the children hold Q from THEIR point of view, so it is
        // negated and weighted by visits (same convention as selfplay.py)
        double tot = 0.0, acc = 0.0;
        for (const auto& [mv, st] : children) { tot += st.N; acc += st.N * (-st.Q); }
        root_values.push_back(tot > 0.0 ? acc / tot : 0.0);

        // how much the search TELLS the moves APART: if this stays small while
        // the entropy is high, the limit is the value head and not the search
        // (see selfplay.py:_root_q_spread for the full reasoning)
        double qmin = 1e30, qmax = -1e30;
        int nq = 0;
        for (const auto& [mv, st] : children)
            if (st.N > 0) { double q = -st.Q; qmin = std::min(qmin, q); qmax = std::max(qmax, q); nq++; }
        if (nq >= 2) res.q_spread_sum += (qmax - qmin);

        float temp = (res.plies < cfg.temp_moves) ? 1.0f : 0.0f;
        Move chosen = select_move(children, temp, rng);
        if (chosen.is_capture()) res.captures += chosen.n_cap;
        if (chosen.promotes) res.promotions++;
        pos = pos.play(chosen);
        res.plies++;
    }

    auto r = pos.result();
    res.truncated = !r.has_value();
    res.no_progress = r.has_value() && pos.no_progress >= NO_PROGRESS_DRAW;
    res.z_white = r.has_value() ? *r : material_result(pos.board);  // truncated: by material
    // d = plies left to the end; the last decision keeps full credit (d = 0)
    // and the earlier ones are shrunk going backwards.
    const size_t n = res.samples.size();
    for (size_t i = 0; i < n; i++) {
        float weight = 1.0f;
        if (cfg.value_discount != 1.0f)
            weight = std::pow(cfg.value_discount, (float)(n - 1 - i));
        res.samples[i].z = (float)(res.z_white * turns[i]) * weight;
    }
    for (size_t i = 0; i < root_values.size(); i++)
        res.value_abs_err += std::fabs(root_values[i] - (double)(res.z_white * turns[i]));
    return res;
}

// ===================== MULTI-THREAD driver =====================
//
// N threads play games in parallel. Each thread has its OWN evaluator (hence
// its own request towards the hub) and its own MCTS tree: no synchronization on
// the tree, the only shared point is the hub's queue.

template <typename MakeEvaluator>
inline std::vector<GameResult> parallel_selfplay(
        int n_games, int n_threads, const SelfPlayConfig& cfg,
        unsigned base_seed, MakeEvaluator make_evaluator,
        bool verbose = false) {
    std::vector<GameResult> results(n_games);
    std::atomic<int> next{0};
    std::atomic<int> done{0};

    auto worker = [&]() {
        auto ev = make_evaluator();          // per-thread evaluator
        for (;;) {
            int g = next.fetch_add(1);
            if (g >= n_games) break;
            results[g] = play_game(*ev, cfg, base_seed + (unsigned)g);
            int d = ++done;
            if (verbose && (d % 10 == 0 || d == n_games))
                printf("  games completed: %d/%d\n", d, n_games);
        }
    };

    std::vector<std::thread> ts;
    ts.reserve(n_threads);
    for (int t = 0; t < n_threads; t++) ts.emplace_back(worker);
    for (auto& t : ts) t.join();
    return results;
}

}  // namespace dama
