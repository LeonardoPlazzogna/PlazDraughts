// Test of the complete MULTI-THREAD self-play (selfplay.hpp + inference_hub.hpp
// + dataset.hpp), with the fake backend: no LibTorch dependency.
//
// It checks the whole chain that runs in production -- N threads, independent
// MCTS trees, a hub merging the evaluations, samples collected and written to a
// file -- against the invariants training relies on:
//   * every sample has pi summing to 1 and a non-empty mask
//   * pi is zero where mask is zero (no probability on illegal actions)
//   * z in {-1, 0, +1}
//   * the written dataset has exactly the expected size
#include "selfplay.hpp"
#include "inference_hub.hpp"
#include "nn_backend.hpp"
#include "dataset.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

using namespace dama;

int main(int argc, char** argv) {
    int n_games = (argc > 1) ? std::atoi(argv[1]) : 8;
    int n_threads = (argc > 2) ? std::atoi(argv[2]) : 4;
    int n_sims = (argc > 3) ? std::atoi(argv[3]) : 60;
    const char* out_path = (argc > 4) ? argv[4] : "selfplay_test.bin";

    FakeBackend backend;
    InferenceHub hub(backend, 256);

    SelfPlayConfig cfg;
    cfg.n_sims = n_sims;
    cfg.batch_size = 8;
    cfg.max_plies = 300;

    printf("self-play: %d games on %d threads, %d simulations/move\n",
           n_games, n_threads, n_sims);

    auto t0 = std::chrono::steady_clock::now();
    auto games = parallel_selfplay(
        n_games, n_threads, cfg, /*base_seed=*/12345,
        [&hub] { return std::make_unique<HubEvaluator>(hub); });
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    hub.stop();

    long long total_samples = 0, total_plies = 0;
    int w = 0, l = 0, d = 0;
    bool ok = true;
    for (int g = 0; g < n_games; g++) {
        const auto& gr = games[g];
        total_samples += (long long)gr.samples.size();
        total_plies += gr.plies;
        if (gr.z_white > 0) w++; else if (gr.z_white < 0) l++; else d++;

        for (const auto& s : gr.samples) {
            double pi_sum = 0.0, mask_sum = 0.0;
            for (int i = 0; i < ENC_POLICY_SIZE; i++) {
                pi_sum += s.pi[i];
                mask_sum += s.mask[i];
                if (s.mask[i] == 0.0f && s.pi[i] != 0.0f) {
                    printf("  ERROR: probability on an illegal action (game %d)\n", g);
                    ok = false;
                }
            }
            if (std::fabs(pi_sum - 1.0) > 1e-4) {
                printf("  ERROR: pi sums to %.6f, not 1 (game %d)\n", pi_sum, g);
                ok = false;
            }
            if (mask_sum < 1.0) {
                printf("  ERROR: empty mask (game %d)\n", g);
                ok = false;
            }
            if (s.z != -1.0f && s.z != 0.0f && s.z != 1.0f) {
                printf("  ERROR: z=%f outside {-1,0,1} (game %d)\n", s.z, g);
                ok = false;
            }
        }
        if (!ok) break;
    }

    printf("  games: White %d  Black %d  draws %d\n", w, l, d);
    printf("  total plies: %lld (mean %.1f)\n", total_plies,
           (double)total_plies / n_games);
    printf("  samples: %lld\n", total_samples);
    printf("  time: %.2fs (%.2f games/s)\n", secs, n_games / secs);
    printf("  hub batches: %lld, mean %.2f positions\n",
           hub.batches(), hub.avg_batch());

    if (!write_dataset(out_path, games)) {
        printf("  ERROR: failed to write the dataset\n");
        ok = false;
    } else {
        // expected size: header (4+4*4) + n*(x + pi + mask + z)
        long long expect = 20 + total_samples *
            (long long)((PLANE_FLOATS + 2 * ENC_POLICY_SIZE + 1) * sizeof(float));
        FILE* f = std::fopen(out_path, "rb");
        std::fseek(f, 0, SEEK_END);
        long long got = std::ftell(f);
        std::fclose(f);
        printf("  dataset: %s (%lld bytes, expected %lld)\n", out_path, got, expect);
        if (got != expect) {
            printf("  ERROR: unexpected dataset size\n");
            ok = false;
        }
    }

    printf("\n%s\n", ok ? "SELF-PLAY MULTI-THREAD OK: sample invariants hold."
                        : "FAILED: see the errors above.");
    return ok ? 0 : 1;
}
