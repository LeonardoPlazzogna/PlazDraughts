// Batched inference hub: the piece that makes a GPU worthwhile.
//
// N self-play threads, each with its own MCTS tree, submit small groups of
// positions to evaluate (the MCTS leaf batching, ~8 positions). A DISPATCHER
// thread merges them into one large batch and calls the backend ONCE. So the
// CPU work (move generation, tree, encoding) scales over the cores and the GPU
// gets batches large enough to be really used.
//
//   thread_1 ... thread_N  --(queue)-->  DISPATCHER  --(1 forward)-->  BACKEND
//        ^                                   |
//        +------(response to the requester)--+
//
// Compared with the Python version (inference_server.py) there is no IPC here:
// the threads share memory, so the request is a pointer and the response is a
// direct write into the caller's buffer. That is why the C++ engine can go far
// beyond the Python pipeline.
//
// NOTE: the backend is an abstraction (nn_backend.hpp), so ALL of this file
// compiles and can be tested without LibTorch (see threading_test.cpp).
#pragma once
#include "mcts.hpp"
#include "encoder.hpp"
#include "nn_backend.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <deque>
#include <vector>
#include <atomic>
#include <memory>

namespace dama {

class InferenceHub;

// Evaluation request. It lives in the calling thread and is REUSED at every
// call (no allocations on the hot path): the dispatcher writes the results
// directly into the requester's buffers.
struct InferRequest {
    const float* planes = nullptr;   // [n * 7*8*8], owned by the caller
    int n = 0;
    float* out_logits = nullptr;     // [n * 256], the caller's
    float* out_values = nullptr;     // [n], the caller's

    std::mutex m;
    std::condition_variable cv;
    bool ready = false;
};

class InferenceHub {
public:
    InferenceHub(NNBackend& backend, int max_batch = 256)
        : backend_(backend), max_batch_(max_batch) {
        dispatcher_ = std::thread([this] { loop(); });
    }

    ~InferenceHub() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(qm_);
            if (stopping_) return;
            stopping_ = true;
        }
        qcv_.notify_all();
        if (dispatcher_.joinable()) dispatcher_.join();
    }

    // Submits and WAITS. Called by the self-play threads.
    void submit_and_wait(InferRequest& req) {
        {
            std::lock_guard<std::mutex> lk(req.m);
            req.ready = false;
        }
        {
            std::lock_guard<std::mutex> lk(qm_);
            queue_.push_back(&req);
        }
        qcv_.notify_one();
        std::unique_lock<std::mutex> lk(req.m);
        req.cv.wait(lk, [&req] { return req.ready; });
    }

    // statistics (read at the end of the run, not on the hot path)
    long long batches() const { return batches_; }
    long long positions() const { return positions_; }
    double avg_batch() const {
        return batches_ ? (double)positions_ / (double)batches_ : 0.0;
    }

private:
    void loop() {
        const int P = ENC_IN_PLANES * 8 * 8;
        std::vector<InferRequest*> batch;
        std::vector<float> planes_buf, logits_buf, values_buf;

        for (;;) {
            batch.clear();
            {
                std::unique_lock<std::mutex> lk(qm_);
                qcv_.wait(lk, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                // Merge: take everything there is, up to max_batch positions in
                // total (positions, not requests: the number of positions sets
                // the size of the forward pass).
                int total = 0;
                while (!queue_.empty()) {
                    InferRequest* r = queue_.front();
                    if (total != 0 && total + r->n > max_batch_) break;
                    queue_.pop_front();
                    batch.push_back(r);
                    total += r->n;
                }
            }
            if (batch.empty()) continue;

            int total = 0;
            for (auto* r : batch) total += r->n;
            planes_buf.resize((size_t)total * P);
            logits_buf.resize((size_t)total * ENC_POLICY_SIZE);
            values_buf.resize((size_t)total);

            size_t off = 0;
            for (auto* r : batch) {
                std::copy(r->planes, r->planes + (size_t)r->n * P,
                          planes_buf.begin() + off * P);
                off += r->n;
            }

            backend_.forward(planes_buf.data(), total,
                             logits_buf.data(), values_buf.data());
            batches_++;
            positions_ += total;

            // scatter: every requester gets EXACTLY its own slice
            off = 0;
            for (auto* r : batch) {
                std::copy(logits_buf.begin() + off * ENC_POLICY_SIZE,
                          logits_buf.begin() + (off + r->n) * ENC_POLICY_SIZE,
                          r->out_logits);
                std::copy(values_buf.begin() + off,
                          values_buf.begin() + off + r->n,
                          r->out_values);
                off += r->n;
                {
                    std::lock_guard<std::mutex> lk(r->m);
                    r->ready = true;
                }
                r->cv.notify_one();
            }
        }
    }

    NNBackend& backend_;
    int max_batch_;
    std::thread dispatcher_;
    std::mutex qm_;
    std::condition_variable qcv_;
    std::deque<InferRequest*> queue_;
    bool stopping_ = false;
    std::atomic<long long> batches_{0};
    std::atomic<long long> positions_{0};
};

// Per-thread evaluator that talks to the hub. It implements IEvaluator, so the
// MCTS (mcts.hpp) does not even know there is a network on the other side: it
// is the same interface UniformEvaluator uses in the tests.
class HubEvaluator : public IEvaluator {
public:
    explicit HubEvaluator(InferenceHub& hub) : hub_(hub) {}

    std::vector<EvalResult> evaluate_batch(const std::vector<Position>& positions) override {
        const int n = (int)positions.size();
        std::vector<EvalResult> out;
        if (n == 0) return out;

        const int P = ENC_IN_PLANES * 8 * 8;
        planes_.resize((size_t)n * P);
        logits_.resize((size_t)n * ENC_POLICY_SIZE);
        values_.resize((size_t)n);

        for (int i = 0; i < n; i++) {
            Planes pl = encode(positions[i].board, positions[i].turn,
                               positions[i].no_progress, NO_PROGRESS_DRAW);
            std::copy(pl.begin(), pl.end(), planes_.begin() + (size_t)i * P);
        }

        req_.planes = planes_.data();
        req_.n = n;
        req_.out_logits = logits_.data();
        req_.out_values = values_.data();
        hub_.submit_and_wait(req_);

        out.reserve(n);
        for (int i = 0; i < n; i++)
            out.push_back(priors_from_logits(positions[i],
                                             logits_.data() + (size_t)i * ENC_POLICY_SIZE,
                                             values_[i]));
        return out;
    }

    // Raw logits -> priors over the legal moves only (summing to 1).
    // Faithful port of evaluators.py:priors_from_logits, including the handling
    // of index collisions: softmax over the DISTINCT indices only, then the
    // probability is split evenly among the moves sharing the index.
    static EvalResult priors_from_logits(const Position& pos,
                                         const float* logits, float value) {
        EvalResult r;
        r.value = value;
        const auto& moves = pos.legal_moves();
        if (moves.empty()) return r;

        bool flip = (pos.turn == BLACK);
        std::vector<int> idx(moves.size());
        for (size_t i = 0; i < moves.size(); i++)
            idx[i] = move_policy_index(moves[i], flip);

        std::vector<int> uniq;                    // distinct, in order
        for (int ix : idx)
            if (std::find(uniq.begin(), uniq.end(), ix) == uniq.end())
                uniq.push_back(ix);

        double mx = -1e30;
        for (int ix : uniq) mx = std::max(mx, (double)logits[ix]);
        double sum = 0.0;
        std::vector<double> ex(uniq.size());
        for (size_t k = 0; k < uniq.size(); k++) {
            ex[k] = std::exp((double)logits[uniq[k]] - mx);
            sum += ex[k];
        }

        r.priors.resize(moves.size());
        for (size_t i = 0; i < moves.size(); i++) {
            size_t k = std::find(uniq.begin(), uniq.end(), idx[i]) - uniq.begin();
            int count = 0;
            for (int ix : idx) if (ix == idx[i]) count++;
            r.priors[i] = (float)(ex[k] / sum / count);
        }
        return r;
    }

private:
    InferenceHub& hub_;
    InferRequest req_;
    std::vector<float> planes_, logits_, values_;
};

}  // namespace dama
