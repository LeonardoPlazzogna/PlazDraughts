// Test of the C++ arena (arena.hpp) with the fake backend.
//
// The most important check is SYMMETRY: with two identical evaluators the
// score must be ~0.5. If the color assignment were wrong -- for instance A
// always White, or the condition inverted -- the score would collapse towards 0
// or 1, because in draughts White moves first and the advantage would all go
// to one side. It is the cheapest way to catch an error that would otherwise
// distort every promotion.
#include "arena.hpp"
#include "inference_hub.hpp"
#include "nn_backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

using namespace dama;

static bool ok = true;

static void check(bool cond, const char* msg) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) ok = false;
}

int main(int argc, char** argv) {
    int n_games = (argc > 1) ? std::atoi(argv[1]) : 20;
    int n_threads = (argc > 2) ? std::atoi(argv[2]) : 4;
    int n_sims = (argc > 3) ? std::atoi(argv[3]) : 40;

    ArenaConfig cfg;
    cfg.n_sims = n_sims;

    printf("arena: %d games on %d threads, %d simulations/move\n",
           n_games, n_threads, n_sims);

    // --- two IDENTICAL backends: the only difference between the sides is the color ---
    FakeBackend ba, bb;
    InferenceHub hub_a(ba, 128), hub_b(bb, 128);

    ArenaResult r = parallel_arena(
        n_games, n_threads, cfg, 12345,
        [&hub_a] { return std::make_unique<HubEvaluator>(hub_a); },
        [&hub_b] { return std::make_unique<HubEvaluator>(hub_b); });

    printf("  A: %d wins  %d draws  %d losses  -> score %.3f\n",
           r.wins, r.draws, r.losses, r.score_a);

    check(r.games == n_games, "games played = games requested");
    check(r.wins + r.draws + r.losses == n_games,
          "wins + draws + losses = games");
    check(r.score_a >= 0.0 && r.score_a <= 1.0, "score in [0,1]");

    // Symmetry: identical evaluators -> neither side is favored.
    // NOTE: on its own this check is weak -- if every game ends in a draw the
    // score is 0.5 anyway, even with a wrong color assignment. The exact check
    // further down is needed.
    check(std::fabs(r.score_a - 0.5) <= 0.25,
          "score ~0.5 with identical evaluators (necessary, not sufficient)");

    // --- determinism: same seed -> same result ---
    ArenaResult r2 = parallel_arena(
        n_games, n_threads, cfg, 12345,
        [&hub_a] { return std::make_unique<HubEvaluator>(hub_a); },
        [&hub_b] { return std::make_unique<HubEvaluator>(hub_b); });
    check(r2.score_a == r.score_a && r2.wins == r.wins && r2.draws == r.draws,
          "same seed -> same result (no dependence on thread order)");

    // --- a different seed must be able to change the outcome ---
    ArenaResult r3 = parallel_arena(
        n_games, n_threads, cfg, 999,
        [&hub_a] { return std::make_unique<HubEvaluator>(hub_a); },
        [&hub_b] { return std::make_unique<HubEvaluator>(hub_b); });
    check(r3.score_a >= 0.0 && r3.score_a <= 1.0, "different seed: valid score");

    // --- zero games must not divide by zero ---
    ArenaResult r0 = parallel_arena(
        0, 2, cfg, 1,
        [&hub_a] { return std::make_unique<HubEvaluator>(hub_a); },
        [&hub_b] { return std::make_unique<HubEvaluator>(hub_b); });
    check(r0.games == 0 && r0.score_a == 0.0, "0 games: no division by zero");

    // ================= EXACT check of the perspective =================
    // Two DIFFERENT players (different bias) and the very same game seen from
    // both sides: X as White against Y, and the same game described as "Y is A
    // and plays Black". Same seed, same colors, same moves: the scores must add
    // up to EXACTLY 1. If the color or perspective logic were wrong, the
    // identity would break on every decisive game (on a draw 0.5 + 0.5 = 1
    // holds anyway, hence the search for decisive games below).
    FakeBackend bx(0.0), by(1.7);          // two distinct "players"
    InferenceHub hub_x(bx, 128), hub_y(by, 128);
    HubEvaluator ex(hub_x), ey(hub_y);

    // A range of seeds is swept: with a fake evaluator many games end in a
    // no-progress draw, and on those the identity 0.5+0.5=1 would hold by
    // chance. DECISIVE games are needed for the check to have power, so they
    // are looked for and counted.
    ArenaConfig icfg = cfg;
    icfg.n_sims = 30;              // more decisive games show up at 30 sims
    bool identity_ok = true;
    int decisive = 0, checked = 0;
    for (unsigned s = 1; s <= 40; s++) {
        // the very same game (X as White, same seed), described first with X
        // as player A and then with X as player B
        double as_a = play_arena_game(ex, ey, icfg, /*a_is_white=*/true, s);
        double as_b = play_arena_game(ey, ex, icfg, /*a_is_white=*/false, s);
        if (std::fabs((as_a + as_b) - 1.0) > 1e-9) identity_ok = false;
        checked++;
        if (as_a != 0.5) decisive++;
    }
    printf("  identity checked on %d seeds, %d of them with a decisive outcome\n",
           checked, decisive);
    check(identity_ok, "score(X as A) + score(X as B) = exactly 1");
    check(decisive > 0,
          "at least one decisive game (the check above is not vacuous)");

    // --- per-side c_puct ------------------------------------------------
    // It serves to calibrate the search: the same network on both sides, two
    // exploration constants, and whoever wins has the better search.
    //
    // The invariant that matters is that the default is INERT: explicitly
    // asking for B the same c_puct as A must give a bit-for-bit identical result
    // to asking nothing. Without this check a slip in here would silently
    // change the behavior of EVERY promotion arena, which has no use for this
    // feature.
    {
        ArenaConfig same = icfg;                 // c_puct_b stays at -1 (inert)
        ArenaConfig expl = icfg;
        expl.c_puct_b = icfg.c_puct;             // requested, but equal to A
        bool inert = true, changed = false;
        for (unsigned s = 700; s < 720; s++) {
            if (play_arena_game(ex, ey, same, true, s) !=
                play_arena_game(ex, ey, expl, true, s)) inert = false;
        }
        check(inert, "c_puct_b equal to A changes nothing (inert default)");

        ArenaConfig diff = icfg;
        diff.c_puct_b = icfg.c_puct * 0.1f;      // much lower exploration
        for (unsigned s = 700; s < 760 && !changed; s++) {
            if (play_arena_game(ex, ey, same, true, s) !=
                play_arena_game(ex, ey, diff, true, s)) changed = true;
        }
        check(changed, "a different c_puct_b really changes the games "
                       "(the check above is not vacuous)");
    }

    // --- how many games are REALLY different --------------------------------
    //
    // The arena uses no Dirichlet noise and the evaluators are deterministic:
    // two games that start with the same moves continue identically. The only
    // source of variety is the opening plies sampled from the visits, and if
    // they are few the match replays the same game over and over. The trouble
    // is not wasted time: the confidence interval computed on the number of
    // games played becomes much narrower than it really is, and winners that
    // are noise get declared. Here it is checked that the count of distinct
    // games sees the difference.
    {
        int n = std::max(n_games, 24);
        ArenaConfig greedy = cfg;
        greedy.temp_plies = 0;                   // no sampled opening
        ArenaResult rg = parallel_arena(
            n, n_threads, greedy, 999,
            [&hub_a] { return std::make_unique<HubEvaluator>(hub_a); },
            [&hub_b] { return std::make_unique<HubEvaluator>(hub_b); });

        ArenaConfig varied = cfg;
        varied.temp_plies = 10;
        ArenaResult rv = parallel_arena(
            n, n_threads, varied, 999,
            [&hub_a] { return std::make_unique<HubEvaluator>(hub_a); },
            [&hub_b] { return std::make_unique<HubEvaluator>(hub_b); });

        printf("  distinct games: %d/%d without sampled openings, "
               "%d/%d with 10 plies\n", rg.distinct, n, rv.distinct, n);
        check(rg.distinct <= 2,
              "without sampled openings the games are all the same");
        check(rv.distinct > rg.distinct,
              "sampling the openings increases the distinct games");
        check(rv.distinct >= n / 2,
              "with 10 sampled plies most games differ");
    }

    hub_x.stop();
    hub_y.stop();
    hub_a.stop();
    hub_b.stop();

    printf("\n%s\n", ok ? "ARENA OK: invariants and color symmetry hold."
                        : "ARENA FAILED.");
    return ok ? 0 : 1;
}
