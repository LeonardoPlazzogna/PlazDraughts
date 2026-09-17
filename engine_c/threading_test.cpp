// Stress test of the batched inference hub (inference_hub.hpp).
//
// It is the most important test of the C++ engine: concurrency is where the
// bugs live that reading the code does not reveal. It checks three distinct
// things:
//
//  1) SCATTER CORRECTNESS (the classic bug): with many threads submitting
//     requests of different lengths, each thread must receive EXACTLY the
//     results computed from ITS OWN input. The fake backend is deterministic,
//     so each thread can recompute on its own what it should receive: if the
//     responses reached the wrong requester (cross-talk), the comparison would
//     fail.
//  2) NO DEADLOCKS: if the hub lost a notification or got stuck, the test would
//     never finish -- a hang is the symptom.
//  3) EFFECTIVE BATCHING: it prints the mean batch size. With many threads it
//     must be > 1, otherwise the hub is not merging anything and would be
//     useless on a GPU.
#include "inference_hub.hpp"
#include "nn_backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <cmath>

using namespace dama;

int main(int argc, char** argv) {
    int n_threads = (argc > 1) ? std::atoi(argv[1]) : 8;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 200;
    int max_batch = (argc > 3) ? std::atoi(argv[3]) : 256;

    const int P = ENC_IN_PLANES * 8 * 8;
    FakeBackend backend;
    InferenceHub hub(backend, max_batch);

    std::atomic<long long> mismatches{0};
    std::atomic<long long> checked{0};

    auto worker = [&](int tid) {
        std::mt19937 rng(1234u + tid);
        std::uniform_int_distribution<int> nd(1, 12);   // different lengths
        InferRequest req;
        std::vector<float> planes, logits, values;
        std::vector<float> expect_logits, expect_values;

        for (int it = 0; it < iters; it++) {
            int n = nd(rng);
            planes.assign((size_t)n * P, 0.0f);
            // UNIQUE content per thread+iteration: if another thread received
            // these results, its comparison would fail
            for (int i = 0; i < n; i++)
                for (int k = 0; k < P; k++)
                    planes[(size_t)i * P + k] =
                        (float)((tid * 7919 + it * 104729 + i * 31 + k) % 13) * 0.1f;

            logits.assign((size_t)n * ENC_POLICY_SIZE, 0.0f);
            values.assign((size_t)n, 0.0f);

            req.planes = planes.data();
            req.n = n;
            req.out_logits = logits.data();
            req.out_values = values.data();
            hub.submit_and_wait(req);

            // recompute locally what SHOULD have arrived
            expect_logits.assign((size_t)n * ENC_POLICY_SIZE, 0.0f);
            expect_values.assign((size_t)n, 0.0f);
            FakeBackend ref;
            ref.forward(planes.data(), n, expect_logits.data(), expect_values.data());

            for (size_t k = 0; k < logits.size(); k++)
                if (std::fabs(logits[k] - expect_logits[k]) > 1e-6f) mismatches++;
            for (size_t k = 0; k < values.size(); k++)
                if (std::fabs(values[k] - expect_values[k]) > 1e-6f) mismatches++;
            checked += n;
        }
    };

    printf("hub: %d threads x %d iterations, max batch %d\n", n_threads, iters, max_batch);
    std::vector<std::thread> ts;
    for (int t = 0; t < n_threads; t++) ts.emplace_back(worker, t);
    for (auto& t : ts) t.join();
    hub.stop();

    printf("  positions checked    : %lld\n", (long long)checked);
    printf("  discrepancies        : %lld\n", (long long)mismatches);
    printf("  batches run          : %lld\n", hub.batches());
    printf("  total positions      : %lld\n", hub.positions());
    printf("  mean batch size      : %.2f\n", hub.avg_batch());

    bool ok = (mismatches == 0);
    // with >1 thread the hub must merge: mean > 1 position per batch
    bool batching_ok = (n_threads == 1) || (hub.avg_batch() > 1.0);
    if (!batching_ok) printf("  WARNING: no real merging (mean <= 1).\n");

    printf("\n%s\n", (ok && batching_ok)
           ? "CONCURRENCY OK: no cross-talk, no deadlock, batching active."
           : "FAILED: see discrepancies/batching above.");
    return (ok && batching_ok) ? 0 : 1;
}
