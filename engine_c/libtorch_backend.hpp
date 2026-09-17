// LibTorch backend: the ONLY part of the engine that depends on PyTorch/CUDA.
//
// It is deliberately THIN -- a few dozen lines -- because it is the one part
// the g++ test builds cannot exercise: everything else in engine_c/ (move
// generation, encoder, MCTS, multi-thread batching hub, self-play, dataset) is
// compiled and tested with the fake backend.
//
// BEFORE USING IT TO PRODUCE RESULTS, on the machine with LibTorch:
//   1. cmake -DWITH_LIBTORCH=ON ... && cmake --build .
//   2. ./parity_check <model_jit.pt> <device>     (see parity_check.cpp)
//      -> must match PyTorch within 1e-4 on the same inputs
// Only once (2) is green does a real run make sense.
//
// MODEL CONTRACT. It expects a TorchScript traced from the Python model
// (model.py:PolicyValueNet), returning a tuple of three tensors:
//   policy_logits [B, 256], value [B, 1], wdl_logits [B, 3]
// Only the first two are used: the priors are rebuilt downstream
// (inference_hub.hpp:priors_from_logits), exactly as on the Python side.
// The model is produced by engine_c/tools/export_jit.py.
#pragma once
#include "nn_backend.hpp"
#include <string>
#include <stdexcept>
#include <cstring>      // std::memcpy (used in forward)

#ifdef WITH_LIBTORCH
#include <torch/script.h>
#include <torch/torch.h>

namespace dama {

// Turns off TF32 (10-bit mantissa) for convolutions and matmul, forcing full
// fp32. NOT needed in production -- TF32 is much faster and the difference does
// not affect play -- but needed for the parity check: with TF32 on, C++ and
// Python can differ by ~1e-3 on the logits from arithmetic alone, and that
// noise MASKS the real question, namely whether the two compute the same
// thing. In strict fp32 the residual difference is rounding error, and any
// significant gap is a real bug.
inline void set_tf32(bool enabled) {
    at::globalContext().setAllowTF32CuBLAS(enabled);
    at::globalContext().setAllowTF32CuDNN(enabled);
}

class LibTorchBackend : public NNBackend {
public:
    LibTorchBackend(const std::string& model_path, const std::string& device,
                    bool fp16 = false)
        : device_(device == "cuda" ? torch::kCUDA : torch::kCPU), fp16_(fp16) {
        if (device == "cuda" && !torch::cuda::is_available())
            throw std::runtime_error(
                "device cuda requested but LibTorch sees no GPU");
        module_ = torch::jit::load(model_path, device_);
        module_.eval();
        // ORDER MATTERS: the conversion to half precision must happen BEFORE
        // the freeze. freeze() inlines the weights as constants in the graph,
        // and a later .to(kHalf) would not convert them reliably: the result
        // would be a model with float32 weights fed with float16 inputs, i.e. a
        // type error at the first forward pass.
        if (fp16_ && device_ == torch::kCUDA)
            module_.to(torch::kHalf);
        // freeze: inlines the weights as constants and folds BatchNorm into the
        // convolutions (the equivalent of what evaluators.py does on the Python
        // side).
        try {
            module_ = torch::jit::freeze(module_);
        } catch (const std::exception&) {
            // safe fallback: a model that cannot be frozen is used as it is
        }
    }

    void forward(const float* planes, int n,
                 float* out_logits, float* out_values) override {
        torch::NoGradGuard no_grad;
        const int P = ENC_IN_PLANES * 8 * 8;

        // from_blob does NOT copy: the buffer belongs to the caller (the hub)
        // and stays valid for the whole call.
        auto input = torch::from_blob(const_cast<float*>(planes),
                                      {n, ENC_IN_PLANES, 8, 8},
                                      torch::kFloat32);
        input = input.to(device_);
        if (fp16_ && device_ == torch::kCUDA) input = input.to(torch::kHalf);

        auto out = module_.forward({input}).toTuple();
        auto logits = out->elements()[0].toTensor().to(torch::kFloat32).cpu().contiguous();
        auto values = out->elements()[1].toTensor().to(torch::kFloat32).cpu().contiguous();

        std::memcpy(out_logits, logits.data_ptr<float>(),
                    (size_t)n * ENC_POLICY_SIZE * sizeof(float));
        // value has shape [n,1]: it is read as n contiguous scalars
        std::memcpy(out_values, values.data_ptr<float>(), (size_t)n * sizeof(float));
    }

    const char* name() const override { return "libtorch"; }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    bool fp16_;
};

}  // namespace dama

#else   // !WITH_LIBTORCH

namespace dama {

// Compilable placeholder: if the engine is built without LibTorch, trying to use
// this backend fails with an explicit message instead of not compiling at all
// (so the rest of the engine stays buildable everywhere).
inline void set_tf32(bool) {}       // without LibTorch there is nothing to adjust

class LibTorchBackend : public NNBackend {
public:
    LibTorchBackend(const std::string&, const std::string&, bool = false) {
        throw std::runtime_error(
            "this executable was built WITHOUT LibTorch: "
            "rebuild with -DWITH_LIBTORCH=ON to use a neural network");
    }
    void forward(const float*, int, float*, float*) override {}
    const char* name() const override { return "libtorch (not compiled)"; }
};

}  // namespace dama

#endif  // WITH_LIBTORCH
