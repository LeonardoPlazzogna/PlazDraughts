# Italian draughts engine — C++ version (`engine_c/`)

Multi-thread self-play engine with batched inference, written to **really use a
GPU**: the bottleneck of the Python pipeline is not the network (a small one)
but the Python overhead on move generation, MCTS and encoding, plus the cost of
IPC between processes. Here the workers are threads in the same process: an
evaluation request is a pointer, the answer a write to shared memory.

```
thread_1 ... thread_N  --(queue)-->  DISPATCHER  --(1 forward)-->  BACKEND
     ^                                   |                       (LibTorch/GPU)
     +--------(answer to the caller)-----+
```

## Status

| Component | File | Status |
|---|---|---|
| Rules and move generation | `dama.hpp` | **validated** — perft 1-7 identical to Python |
| State/transitions | `position.hpp` | **validated** — used by every test |
| Encoder + compact action | `encoder.hpp` | **validated** — numerical parity with Python |
| MCTS tree | `mcts.hpp` | **validated** — 75+ games, count invariants |
| Multi-thread batching hub | `inference_hub.hpp` | **validated** — stress test up to 32 threads |
| Self-play + parallel driver | `selfplay.hpp` | **validated** — sample invariants |
| Dataset writer | `dataset.hpp` | **validated** — round-trip into the Python training |
| Abstract + fake backend | `nn_backend.hpp` | **validated** |
| **LibTorch backend** | `libtorch_backend.hpp` | **validated on Windows/MSVC** — parity 0.000e+00 |
| Executable | `main.cpp` | **validated** — self-play with the real network, dataset read back by Python |
| Parity check | `parity_check.cpp` | **run**: 0.000e+00 against PyTorch |

## Build and validation

Without LibTorch — no external dependency, everything compiles and the tests run:

```bash
./build.sh            # Linux/macOS   (.\build.ps1 on Windows)
```

It builds and runs in sequence: `perft`, `encoder_parity`, `mcts_selfplay_test`,
`threading_test`, `selfplay_mt_test`, `arena_test`, plus the `dama_engine`
executable.

With CMake (equivalent, plus `ctest`):

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ctest --test-dir build --output-on-failure
```

With LibTorch, to use a real network and the GPU:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DWITH_LIBTORCH=ON \
      -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
```

On Windows `build.ps1` does the same, adding the two steps that are needed there
and not elsewhere — `--config Release` (the generator is multi-configuration:
without it, the build is Debug) and the check that the binary **really starts**,
because a LibTorch executable missing its DLLs dies before `main` without
printing anything:

```powershell
.\build.ps1 -LibTorch C:\path\to\libtorch
```

It needs LibTorch in its Windows variant and Visual Studio with the C++ tools:
MinGW will not do, those libraries are built with MSVC. Without `-LibTorch` the
script builds with g++, runs all the tests and warns that the engine produced
has **only the fake backend**.

## Usage

From the project root, with the engine built with LibTorch:

```bash
# generates self-play games and writes the dataset for the Python training
./engine_c/build/dama_engine --weights model_jit.pt --device cuda --n-games 200 \
                             --n-threads 16 --n-sims 400 --out dataset.bin
```

The TorchScript model is exported from the Python network:

```bash
python engine_c/tools/export_jit.py weights/champion.pt model_jit.pt
```

The dataset produced is read back by training with `cpp_dataset.load_dataset()`,
in the same form `train.train_on_samples` already expects: **training stays in
PyTorch/Python, unchanged**.

## The LibTorch backend, verified on 19 August 2026

For a long time this section said that the backend was not verified, because
LibTorch and CMake were missing. They were less missing than it seemed: the
Visual Studio build tools already include CMake and Ninja, and LibTorch for
Windows in its CPU-only variant weighs 227 MB. With those, the chain closes:

| check | outcome |
|---|---|
| build with MSVC 19.50 | no errors, 7 executables |
| `ctest` | **6 tests out of 6** |
| `parity_check` against PyTorch | **0.000e+00** on logits, probabilities and values |
| negative control of the same test | a wrong output would give 6075 times the threshold |
| self-play with the real network | 4 games, 821 samples, 205 plies on average |
| dataset read back by the Python training | passed |
| conductor using the engine | 2 complete cycles, self-play and arena |

**What changes in practice on Windows**: with the compiled engine the conductor
no longer starts the pool of Python processes, and with it both the cap of four
processes and the native crash that imposed it go away.

The **CUDA variant** stays unverified on this machine, which has no NVIDIA card.
The procedure is identical, only the LibTorch archive changes.

The backend stays deliberately **thin** anyway (a few dozen lines: load the
model, one forward pass, copy the results) and sits behind an abstraction: all
the hard code — concurrency, batching, MCTS, self-play — is tested with a
deterministic fake backend and **does not change** when LibTorch is linked.

The branch without LibTorch still compiles and raises an explicit error if one
tries to use it, so the engine stays buildable everywhere.

**Before using the engine to produce results**, on the machine with LibTorch,
from the project root:

```bash
python engine_c/tools/export_jit.py weights/champion.pt model_jit.pt cuda
./engine_c/build/parity_check model_jit.pt cuda
```

It must print `PARITY OK` (difference < 1e-4 from PyTorch on the same inputs).
The last argument of the export is the device the reference is computed on, and
it must be the one the check runs on: CPU and CUDA differ by ~1e-3 on their own.
It is the make-or-break test: the risk is not that the engine is slow, it is
that it plays in a subtly different way without anyone noticing.

Also recommended, on Linux where the sanitizers work (on MSYS2/Windows they do
not link):

```bash
g++ -std=c++20 -O1 -g -fsanitize=thread -pthread threading_test.cpp -o tsan_test && ./tsan_test 8 100 256
```

## Real bugs found by testing

They are not hypotheses: both came out of running the tests, not reading the code.

1. **Use-after-free in `mcts.hpp`** — `expand()` received `moves`/`priors` by
   reference, but that reference could point inside `pos_[id]` itself;
   `new_node()` does a `push_back` on `pos_` and can reallocate it, invalidating
   the reference halfway through the function. The self-play test made the
   binary segfault at its first launch. Fixed with a defensive copy.
2. **Performance regression in `has_capture`** — it called `gen_captures` in
   full (all the chains) just to answer yes/no, while the Python version does a
   light check with early exit. It was called at every `encode()`, that is at
   every MCTS leaf.

## Measured performance (fake backend, development machine)

With the fake backend — so **only** move generation + MCTS + encoding, without
the cost of a real network:

- self-play at 50 simulations/move, 4 threads: ~30 games/s
- hub: mean batch of 10 positions with 4 threads, **145 with 32 threads**

The last one is the number that matters for the GPU: the more concurrent
threads, the larger the batches the card receives at once. The real throughput
on a GPU depends on the cost of the network, which cannot be measured here.
