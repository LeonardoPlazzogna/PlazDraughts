// ARENA: two networks play each other; the winner becomes the new champion.
// Faithful port of parallel.py:play_arena_game / WorkerPool.arena.
//
// Deliberate differences from self-play (selfplay.hpp), all for consistency
// with the Python behavior this phase must reproduce:
//
//   * NO Dirichlet noise at the root, and temperature only for the first
//     `temp_plies` plies (see ArenaConfig); after that both sides play greedy
//     (most visited move). The arena measures strength, it does not generate
//     data: more exploration would distort the comparison.
//   * Truncation at max_plies counts as a DRAW, not as a material
//     adjudication. Self-play adjudicates by material because there a learning
//     signal is needed even from an interrupted game; here adjudicating by
//     material would change the promotions compared with the Python pipeline.
//   * The two sides use DIFFERENT evaluators (two networks), alternating White
//     by game index, exactly as the Python conductor does.
#pragma once
#include "position.hpp"
#include "mcts.hpp"
#include "selfplay.hpp"     // select_move (sampling with temperature)
#include <vector>
#include <random>
#include <atomic>
#include <thread>
#include <algorithm>
#include <unordered_set>
#include <cstdint>

namespace dama {

struct ArenaConfig {
    int n_sims = 400;
    float c_puct = 1.5f;
    // Exploration of side B. Negative = "use the same value as A", which is the
    // normal case (promotion arena: two networks, same search).
    //
    // It serves to measure the SEARCH instead of the network: with the very
    // same network on both sides and two different c_puct values, whoever wins
    // has the better search. It is the only way to answer the question left
    // open by the analysis of the run -- the sharpness of the policy target can
    // already be measured, the PLAYING STRENGTH cannot, and the two do not
    // coincide.
    float c_puct_b = -1.0f;
    // Simulations of side B. Negative = same as A (the normal case).
    //
    // It answers a budget question that observation alone does not settle:
    // whether doubling the simulations is worth more than the half of the
    // games that the doubling costs. Same network on both sides, different
    // budgets, and the score says how much the extra search is worth.
    int n_sims_b = -1;
    // Leaf batching of side B. Negative = same as A.
    //
    // It answers the question the normal arena CANNOT ask: does more batching
    // make the search worse? With the same network on both sides and two
    // different batch sizes, whoever wins has the better search. In the
    // promotion arena both sides use the same value, so any degradation cancels
    // out on both sides and the score stays at 0.5: the defect would be
    // invisible exactly to the measurement that should detect it.
    int batch_size_b = -1;
    int batch_size = 8;
    int max_plies = 300;
    // Opening plies played by sampling from the visits instead of greedy.
    //
    // Without this the arena is COMPLETELY DETERMINISTIC: no Dirichlet noise +
    // a deterministic network = every game with the same colors is the very
    // same game, and playing 30 of them gives exactly the same information as
    // playing 2 (one per color). Also verified experimentally on the Python
    // arena, which had the same defect: four different seeds produced the very
    // same result.
    //
    // A few sampled plies are enough to diversify the openings without
    // distorting the strength measurement, and since they apply to both sides
    // they favor neither. It is the equivalent of the opening book the chess
    // project uses for the same purpose. The conductor passes
    // DAMA_ARENA_TEMP_PLIES (10 by default); 4 is only the default for direct use.
    int temp_plies = 4;
};

// One game A vs B. Returns A's score: 1.0 win, 0.5 draw, 0.0 loss. `a_is_white`
// decides which side A plays.
// `out_sig`, if given, receives a fingerprint of the sequence of moves played.
// It serves to count how many DIFFERENT games a match really contains: without
// Dirichlet noise and with deterministic networks, two games that start the
// same continue the same, so the number of games played and the number of
// independent games do not coincide, and confidence intervals are computed on
// the latter.
inline double play_arena_game(IEvaluator& eval_a, IEvaluator& eval_b,
                              const ArenaConfig& cfg, bool a_is_white,
                              unsigned seed, uint64_t* out_sig = nullptr,
                              int* out_plies = nullptr) {
    std::mt19937 rng(seed);
    MCTSConfig mc_a;
    mc_a.n_sims = cfg.n_sims;
    mc_a.c_puct = cfg.c_puct;
    mc_a.batch_size = cfg.batch_size;
    // Side B configuration: identical to A except for the parameters that were
    // explicitly requested.
    MCTSConfig mc_b = mc_a;
    if (cfg.c_puct_b >= 0.0f) mc_b.c_puct = cfg.c_puct_b;
    if (cfg.n_sims_b > 0)     mc_b.n_sims = cfg.n_sims_b;
    if (cfg.batch_size_b > 0) mc_b.batch_size = cfg.batch_size_b;

    Position pos;
    int plies = 0;
    while (!pos.is_terminal() && plies < cfg.max_plies) {
        // the side to move is A if (White to move) == (A plays White)
        bool a_to_move = ((pos.turn == WHITE) == a_is_white);
        IEvaluator& ev = a_to_move ? eval_a : eval_b;
        const MCTSConfig& mc = a_to_move ? mc_a : mc_b;

        MCTS mcts(ev, mc, rng);
        auto children = mcts.run(pos, /*add_noise=*/false);
        if (children.empty()) break;

        // Sampled opening (diversifies the games), then greedy: from ply
        // temp_plies on the most visited move always wins.
        float temp = (plies < cfg.temp_plies) ? 1.0f : 0.0f;
        Move chosen = select_move(children, temp, rng);
        if (out_sig) {   // FNV-1a over the move (from, to, captures)
            *out_sig ^= (uint64_t)(chosen.frm * 64 + chosen.to
                                   + chosen.n_cap * 4096);
            *out_sig *= 1099511628211ull;
        }
        pos = pos.play(chosen);
        plies++;
    }

    // The LENGTH serves whoever compares two search configurations: changing
    // the leaf batching changes the games a little, and without this number
    // "games per hour" would mix the machine's speed with how much work there
    // was to do.
    if (out_plies) *out_plies = plies;

    // Truncation -> draw (0), like "pos.result() or 0" on the Python side.
    auto r = pos.result();
    int z_white = r.has_value() ? *r : 0;
    int a_res = a_is_white ? z_white : -z_white;
    return (a_res > 0) ? 1.0 : ((a_res == 0) ? 0.5 : 0.0);
}

struct ArenaResult {
    double score_a = 0.0;   // in [0,1]
    int wins = 0, draws = 0, losses = 0;   // from A's point of view
    int games = 0;
    // Games with a distinct move sequence. If it is much smaller than `games`,
    // the match replayed the same games several times and its uncertainty is
    // that of `distinct` games, not of `games`.
    int distinct = 0;
    // Plies played in total and mean per game. The cost of a game is
    // proportional to its length, so without this two configurations cannot be
    // compared on time per game.
    long long plies = 0;
    double plies_mean = 0.0;
};

// Multi-thread driver. `make_eval_a`/`make_eval_b` build one evaluator PER
// THREAD (each with its own request towards its hub): two separate hubs are
// needed because the two models are different networks and cannot share a
// batch.
template <typename MakeEvalA, typename MakeEvalB>
inline ArenaResult parallel_arena(int n_games, int n_threads,
                                  const ArenaConfig& cfg, unsigned base_seed,
                                  MakeEvalA make_eval_a, MakeEvalB make_eval_b) {
    std::vector<double> scores(n_games, 0.0);
    std::vector<uint64_t> sigs(n_games, 1469598103934665603ull);
    std::vector<int> plies(n_games, 0);
    std::atomic<int> next{0};

    auto worker = [&]() {
        auto ea = make_eval_a();
        auto eb = make_eval_b();
        for (;;) {
            int g = next.fetch_add(1);
            if (g >= n_games) break;
            // same convention as the Python: A plays White in the even-indexed
            // games, and the seed is base_seed + index
            scores[g] = play_arena_game(*ea, *eb, cfg, (g % 2 == 0),
                                        base_seed + (unsigned)g, &sigs[g],
                                        &plies[g]);
        }
    };

    std::vector<std::thread> ts;
    ts.reserve(n_threads);
    for (int t = 0; t < n_threads; t++) ts.emplace_back(worker);
    for (auto& t : ts) t.join();

    ArenaResult res;
    res.games = n_games;
    for (double s : scores) {
        res.score_a += s;
        if (s > 0.75) res.wins++;
        else if (s < 0.25) res.losses++;
        else res.draws++;
    }
    for (int p : plies) res.plies += p;
    if (n_games > 0) res.score_a /= n_games;
    if (n_games > 0) res.plies_mean = (double)res.plies / n_games;
    // Two games with the same move sequence are the same game, even with
    // different indices: here the really different ones are counted. The color
    // is deliberately NOT part of the fingerprint -- a trajectory already seen
    // brings no new information about the game, from whichever side it is
    // watched.
    res.distinct = (int)std::unordered_set<uint64_t>(sigs.begin(), sigs.end()).size();
    return res;
}

}  // namespace dama
