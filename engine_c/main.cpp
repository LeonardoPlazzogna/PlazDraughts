// Engine executable: plays self-play games in parallel and writes the dataset
// that the Python training reads back (cpp_dataset.py), or plays an arena match
// between two networks (--mode arena).
//
// Typical use on a machine with a GPU:
//   ./dama_engine --weights model_jit.pt --device cuda --n-games 200 --n-threads 16 --n-sims 400 --out dataset.bin
//
// Without --weights it uses a fake backend (no network): only for checking the
// pipeline and measuring the cost of move generation/MCTS on their own.
#include "selfplay.hpp"
#include "arena.hpp"
#include "inference_hub.hpp"
#include "nn_backend.hpp"
#include "libtorch_backend.hpp"
#include "dataset.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <string>

using namespace dama;

static void usage() {
    printf(
        "usage: dama_engine [options]\n"
        "  --mode selfplay|arena mode (default selfplay)\n"
        "  --weights <file.pt>   TorchScript model (without it: fake backend)\n"
        "  --weights-b <file.pt> second model, required in arena mode\n"
        "  --device cpu|cuda     where the network runs (default cpu)\n"
        "  --fp16                half-precision inference (cuda only)\n"
        "  --n-games N           games to play (default 100)\n"
        "  --n-threads N         self-play threads (default 8)\n"
        "  --n-sims N            MCTS simulations per move (default 400)\n"
        "  --c-puct F            PUCT exploration (default 1.5)\n"
        "  --value-discount G    shrinks the value target with the distance\n"
        "                        from the end: z * G^d. 1 = off (default).\n"
        "                        Breaks the tie between moves in a won position\n"
        "  --mcts-batch N        leaf batching per thread (default 8)\n"
        "  --max-batch N         maximum hub batch (default 256)\n"
        "  --c-puct-b F          exploration of side B in arena mode;\n"
        "                        default: same as A. For calibrating the search\n"
        "  --n-sims-b N          simulations of side B in arena mode;\n"
        "                        default: same as A. For calibrating the budget\n"
        "  --mcts-batch-b N      leaf batching of side B in arena mode;\n"
        "                        default: same as A. Tells whether more batching\n"
        "                        weakens the search: the only way to see it,\n"
        "                        because with the same value on both sides any\n"
        "                        degradation cancels out\n"
        "  --temp-plies N        opening plies sampled instead of greedy in\n"
        "                        arena mode (default 4). Raises the number of\n"
        "                        distinct openings: games with the same opening\n"
        "                        are the same game and do not count as new samples\n"
        "  --seed N              base seed (default 1)\n"
        "  --out <file.bin>      output dataset (default selfplay.bin)\n");
}

int main(int argc, char** argv) {
    std::string weights, weights_b, mode = "selfplay", device = "cpu",
                out = "selfplay.bin";
    bool fp16 = false;
    int n_games = 100, n_threads = 8, n_sims = 400, mcts_batch = 8, max_batch = 256;
    float c_puct = 1.5f;
    float value_discount = 1.0f;
    // Negative = side B uses the same c_puct as A (the normal case).
    float c_puct_b = -1.0f;
    int n_sims_b = -1;   // negative = same as A
    int mcts_batch_b = -1;   // negative = same as A
    int temp_plies = -1; // negative = ArenaConfig default
    unsigned seed = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--weights") weights = next();
        else if (a == "--weights-b") weights_b = next();
        else if (a == "--mode") mode = next();
        else if (a == "--device") device = next();
        else if (a == "--fp16") fp16 = true;
        else if (a == "--n-games") n_games = std::atoi(next().c_str());
        else if (a == "--n-threads") n_threads = std::atoi(next().c_str());
        else if (a == "--n-sims") n_sims = std::atoi(next().c_str());
        else if (a == "--c-puct") c_puct = (float)std::atof(next().c_str());
        else if (a == "--value-discount") value_discount = (float)std::atof(next().c_str());
        else if (a == "--c-puct-b") c_puct_b = (float)std::atof(next().c_str());
        else if (a == "--n-sims-b") n_sims_b = std::atoi(next().c_str());
        else if (a == "--mcts-batch-b") mcts_batch_b = std::atoi(next().c_str());
        else if (a == "--temp-plies") temp_plies = std::atoi(next().c_str());
        else if (a == "--mcts-batch") mcts_batch = std::atoi(next().c_str());
        else if (a == "--max-batch") max_batch = std::atoi(next().c_str());
        else if (a == "--seed") seed = (unsigned)std::atoi(next().c_str());
        else if (a == "--out") out = next();
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { printf("unknown option: %s\n", a.c_str()); usage(); return 2; }
    }

    // The device must be validated HERE: the LibTorch backend does
    // (device == "cuda" ? kCUDA : kCPU), so any typo -- "Cuda", "gpu",
    // "cuda:0" -- would silently become CPU, and one would notice that the GPU
    // is not in use only from the slowness.
    if (device != "cpu" && device != "cuda") {
        printf("unknown device: '%s' (allowed values: cpu, cuda)\n",
               device.c_str());
        return 2;
    }
    if (mode != "selfplay" && mode != "arena") {
        printf("unknown mode: '%s' (allowed values: selfplay, arena)\n",
               mode.c_str());
        return 2;
    }

    std::unique_ptr<NNBackend> backend;
    if (weights.empty()) {
        printf("WARNING: no --weights, using the fake backend "
               "(the results have NO playing value).\n");
        backend = std::make_unique<FakeBackend>();
    } else {
        try {
            backend = std::make_unique<LibTorchBackend>(weights, device, fp16);
        } catch (const std::exception& e) {
            printf("error loading the model: %s\n", e.what());
            return 1;
        }
    }

    printf("engine: backend=%s device=%s mode=%s games=%d threads=%d sims=%d "
           "mcts_batch=%d max_batch=%d\n",
           backend->name(), device.c_str(), mode.c_str(), n_games, n_threads,
           n_sims, mcts_batch, max_batch);

    // ===================== ARENA mode =====================
    if (mode == "arena") {
        // Second backend: two different networks cannot be batched together,
        // so each one has its own hub (and its own dispatcher thread).
        std::unique_ptr<NNBackend> backend_b;
        if (weights_b.empty()) {
            if (!weights.empty()) {
                printf("arena mode also requires --weights-b\n");
                return 2;
            }
            backend_b = std::make_unique<FakeBackend>();
        } else {
            try {
                backend_b = std::make_unique<LibTorchBackend>(weights_b, device, fp16);
            } catch (const std::exception& e) {
                printf("error loading model B: %s\n", e.what());
                return 1;
            }
        }

        InferenceHub hub_a(*backend, max_batch);
        InferenceHub hub_b(*backend_b, max_batch);
        ArenaConfig acfg;
        acfg.n_sims = n_sims;
        acfg.batch_size = mcts_batch;
        acfg.c_puct = c_puct;
        acfg.c_puct_b = c_puct_b;
        acfg.n_sims_b = n_sims_b;
        acfg.batch_size_b = mcts_batch_b;
        if (temp_plies >= 0) acfg.temp_plies = temp_plies;

        auto ta0 = std::chrono::steady_clock::now();
        ArenaResult ar = parallel_arena(
            n_games, n_threads, acfg, seed,
            [&hub_a] { return std::make_unique<HubEvaluator>(hub_a); },
            [&hub_b] { return std::make_unique<HubEvaluator>(hub_b); });
        auto ta1 = std::chrono::steady_clock::now();
        double asecs = std::chrono::duration<double>(ta1 - ta0).count();
        hub_a.stop();
        hub_b.stop();

        printf("\ndone in %.1fs\n", asecs);
        printf("  games/hour         : %.0f\n", n_games / asecs * 3600.0);
        printf("  A: %d wins  %d draws  %d losses\n",
               ar.wins, ar.draws, ar.losses);
        printf("  distinct games     : %d of %d\n", ar.distinct, n_games);
        // The arena batches must be reported like the self-play ones, and here
        // they matter more: the arena uses TWO separate hubs -- the networks
        // differ and cannot share a batch -- so each hub sees about half of the
        // requests and the batches reaching the card are about half as large.
        // It is the most expensive phase of the cycle and was the only one
        // without this diagnostic: without it, on a GPU machine there is no way
        // to notice that batching is poor.
        printf("  hub A batches      : %lld (mean %.1f positions)\n",
               hub_a.batches(), hub_a.avg_batch());
        printf("  hub B batches      : %lld (mean %.1f positions)\n",
               hub_b.batches(), hub_b.avg_batch());
        if (ar.distinct * 2 < n_games)
            printf("  WARNING: fewer than half of the games differ from the\n"
                   "  others. The uncertainty of this result is that of %d\n"
                   "  games, not %d: raise --temp-plies to vary the openings.\n",
                   ar.distinct, n_games);
        printf("  plies              : %lld (mean %.1f per game)\n",
               ar.plies, ar.plies_mean);
        // In the same format as self-play, so the Python side reads it with the
        // same code. It serves to compare two search configurations: leaf
        // batching changes the games a little, and without the length the time
        // per game would mix the machine's speed with how much work there was.
        printf("STATS games=%d plies_mean=%.3f\n", n_games, ar.plies_mean);
        // Fixed-format line: the one the Python conductor parses. Besides the
        // score it carries the COUNTS, which the Python SPRT needs: with many
        // draws the real variance is much lower than the binomial one, and
        // estimating it from the counts shortens the matches.
        printf("ARENA_SCORE %.6f wins=%d draws=%d losses=%d distinct=%d\n",
               ar.score_a, ar.wins, ar.draws, ar.losses, ar.distinct);
        return 0;
    }

    InferenceHub hub(*backend, max_batch);
    SelfPlayConfig cfg;
    cfg.n_sims = n_sims;
    cfg.batch_size = mcts_batch;
    cfg.c_puct = c_puct;
    cfg.value_discount = value_discount;

    auto t0 = std::chrono::steady_clock::now();
    auto games = parallel_selfplay(
        n_games, n_threads, cfg, seed,
        [&hub] { return std::make_unique<HubEvaluator>(hub); },
        /*verbose=*/true);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    hub.stop();

    long long samples = 0, plies = 0;
    int w = 0, l = 0, d = 0, trunc = 0, nprog = 0, plies_max = 0;
    long long caps = 0, promos = 0;
    double ent_sum = 0.0, verr_sum = 0.0, qsp_sum = 0.0;
    long long ent_n = 0;
    for (const auto& g : games) {
        samples += (long long)g.samples.size();
        plies += g.plies;
        plies_max = std::max(plies_max, g.plies);
        if (g.z_white > 0) w++; else if (g.z_white < 0) l++; else d++;
        if (g.truncated) trunc++;
        if (g.no_progress) nprog++;
        caps += g.captures;
        promos += g.promotions;
        ent_sum += g.entropy_sum;
        qsp_sum += g.q_spread_sum;
        verr_sum += g.value_abs_err;
        ent_n += (long long)g.samples.size();
    }

    printf("\ndone in %.1fs\n", secs);
    printf("  games/hour         : %.0f\n", n_games / secs * 3600.0);
    printf("  outcomes           : White %d  Black %d  draws %d\n", w, l, d);
    printf("  total plies        : %lld (mean %.1f)\n", plies, (double)plies / n_games);
    printf("  samples            : %lld\n", samples);
    printf("  hub batches        : %lld (mean %.1f positions)\n",
           hub.batches(), hub.avg_batch());

    if (!write_dataset(out, games)) {
        printf("ERROR: failed to write the dataset to %s\n", out.c_str());
        return 1;
    }
    printf("  dataset written    : %s\n", out.c_str());

    // Fixed-format line for the Python conductor (cpp_engine.py): the same
    // diagnostic metrics the Python path collects on its own, so metrics.csv
    // has the same columns filled on both paths.
    printf("STATS games=%d plies_mean=%.3f plies_max=%d white=%d black=%d draws=%d "
           "truncated=%d no_progress=%d captures_mean=%.3f promotions_mean=%.3f "
           "entropy_mean=%.5f q_spread_mean=%.5f value_mae=%.5f\n",
           n_games, (double)plies / std::max(n_games, 1), plies_max, w, l, d,
           trunc, nprog, (double)caps / std::max(n_games, 1),
           (double)promos / std::max(n_games, 1),
           ent_n ? ent_sum / (double)ent_n : 0.0,
           ent_n ? qsp_sum / (double)ent_n : 0.0,
           ent_n ? verr_sum / (double)ent_n : 0.0);
    return 0;
}
