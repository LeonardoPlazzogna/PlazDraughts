// Validates the MCTS tree (position.hpp + mcts.hpp) by playing complete games
// with a UniformEvaluator -- no LibTorch dependency. It checks: no crash over
// many simulations/games, every search produces exactly the expected number of
// root legal moves, the total of the visits to the root's children always
// equals n_sims (the counting invariant of the backup), and games end correctly
// (no legal moves or a no-progress draw) within the safety limit.
#include "position.hpp"
#include "mcts.hpp"
#include <cstdio>
#include <algorithm>

using namespace dama;

int main(int argc, char** argv) {
    unsigned seed = (argc > 1) ? static_cast<unsigned>(std::atoi(argv[1])) : 42u;
    int n_games_arg = (argc > 2) ? std::atoi(argv[2]) : 20;
    int n_sims_arg = (argc > 3) ? std::atoi(argv[3]) : 40;
    int batch_arg = (argc > 4) ? std::atoi(argv[4]) : 4;

    std::mt19937 rng(seed);
    UniformEvaluator ev;
    MCTSConfig cfg;
    cfg.n_sims = n_sims_arg;
    cfg.batch_size = batch_arg;

    const int N_GAMES = n_games_arg;
    const int MAX_PLIES = 300;
    long total_plies = 0;
    int white_wins = 0, black_wins = 0, draws = 0, truncated = 0;
    bool all_ok = true;

    for (int g = 0; g < N_GAMES && all_ok; g++) {
        Position pos;
        int plies = 0;
        while (!pos.is_terminal() && plies < MAX_PLIES) {
            MCTS mcts(ev, cfg, rng);
            auto children = mcts.run(pos, /*add_noise=*/true);

            const auto& legal = pos.legal_moves();
            if (children.size() != legal.size()) {
                fprintf(stderr, "ERROR game %d move %d: %zu root children, expected %zu legal moves\n",
                        g, plies, children.size(), legal.size());
                all_ok = false; break;
            }
            double sum_n = 0.0;
            for (auto& [mv, st] : children) sum_n += st.N;
            if (std::abs(sum_n - cfg.n_sims) > 1e-9) {
                fprintf(stderr, "ERROR game %d move %d: sum of child visits=%.1f, expected %d\n",
                        g, plies, sum_n, cfg.n_sims);
                all_ok = false; break;
            }

            auto best = std::max_element(children.begin(), children.end(),
                [](const auto& a, const auto& b) { return a.second.N < b.second.N; });
            pos = pos.play(best->first);
            plies++;
        }
        if (!all_ok) break;

        total_plies += plies;
        if (plies >= MAX_PLIES) {
            truncated++;
            printf("game %2d: truncated at %d plies (safety limit)\n", g, plies);
        } else {
            auto r = pos.result();
            if (!r.has_value()) {
                fprintf(stderr, "ERROR game %d: is_terminal() true but result() empty\n", g);
                all_ok = false; break;
            }
            if (*r > 0) white_wins++; else if (*r < 0) black_wins++; else draws++;
            printf("game %2d: %3d plies, outcome=%+d\n", g, plies, *r);
        }
    }

    if (!all_ok) { printf("\nVALIDATION FAILED.\n"); return 1; }

    printf("\n%d games, %ld total plies (mean %.1f/game)\n",
           N_GAMES, total_plies, (double)total_plies / N_GAMES);
    printf("White %d  Black %d  Draws %d  Truncated %d\n",
           white_wins, black_wins, draws, truncated);
    printf("\nNO CRASH, NO COUNT DISCREPANCY, ALL GAMES ENDED CORRECTLY.\n");
    return 0;
}
