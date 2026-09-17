// Neural-network backend: the ONLY interface that touches the inference
// framework. The rest of the engine (MCTS, batching, self-play) is written
// against this abstraction, so it compiles and can be tested without LibTorch.
//
// Two implementations:
//   - FakeBackend      (below): deterministic, for the concurrency tests.
//   - LibTorchBackend  (libtorch_backend.hpp): the real one, requires LibTorch.
#pragma once
#include "encoder.hpp"
#include <cstddef>
#include <vector>
#include <cmath>

namespace dama {

class NNBackend {
public:
    virtual ~NNBackend() = default;

    // planes:      [n * ENC_IN_PLANES*8*8] contiguous (n positions)
    // out_logits:  [n * ENC_POLICY_SIZE]   contiguous, to be filled
    // out_values:  [n]                     contiguous, to be filled, in [-1,1]
    // It is called by ONE thread at a time (the hub's dispatcher), so it needs
    // no internal synchronization.
    virtual void forward(const float* planes, int n,
                         float* out_logits, float* out_values) = 0;

    virtual const char* name() const = 0;
};

// DETERMINISTIC fake backend: the output depends reproducibly on the input
// planes only. It serves the concurrency tests: if the scatter of the responses
// were wrong (results delivered to the wrong thread), the expected values would
// not match and the test would fail. It has no playing value: it is not a
// heuristic, it is a hash function.
class FakeBackend : public NNBackend {
public:
    // `bias` shifts the output deterministically: two instances with different
    // biases behave like two distinct "players", useful to test the arena
    // (which must tell the two sides apart) without two real networks.
    explicit FakeBackend(double bias = 0.0) : bias_(bias) {}

    void forward(const float* planes, int n,
                 float* out_logits, float* out_values) override {
        const int P = ENC_IN_PLANES * 8 * 8;
        for (int i = 0; i < n; i++) {
            const float* in = planes + (size_t)i * P;
            double acc = 0.0;
            for (int k = 0; k < P; k++) acc += in[k] * (k + 1);
            for (int a = 0; a < ENC_POLICY_SIZE; a++)
                out_logits[(size_t)i * ENC_POLICY_SIZE + a] =
                    (float)std::sin(acc * 0.001 + a * 0.01 + bias_);
            out_values[i] = (float)std::tanh(acc * 1e-4 + bias_ * 0.1);
        }
        calls++;
        total_positions += n;
    }
    const char* name() const override { return "fake"; }

    // statistics useful to tests/benchmarks: how many calls and how many
    // positions, i.e. how well batching is working.
    long long calls = 0;
    long long total_positions = 0;

private:
    double bias_ = 0.0;
};

}  // namespace dama
