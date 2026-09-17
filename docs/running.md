# Running

How to validate a machine, estimate a run, launch it, follow it and leave it
unattended, on Windows and on Linux. The design behind the pipeline is in
[`design.md`](design.md); the measured results are in [`results.md`](results.md).

## 1. Before a run

### The single command

```powershell
launch.cmd preflight          # Windows
```

```bash
./preflight.sh                # Linux
```

It checks everything in one go and writes a `preflight_report_<date>.txt`: the
environment (PyTorch, CUDA, consistency with `DAMA_DEVICE`), the build and validation
of the C++ engine, every Python test (discovered from `tests/`, not listed by hand),
the cores really available, a benchmark with the estimated duration of the complete
run, a resume test, and whether the engine has the LibTorch backend. It ends with a
single summary line to send to whoever asked for the measurement.

To measure the production configuration, tell it what the run will use:

```powershell
$env:LIBTORCH          = 'C:\libtorch'                        # otherwise the engine gets the fake backend
$env:DAMA_CPP_ENGINE   = 'engine_c\build\dama_engine.exe'
$env:DAMA_DEVICE       = 'cuda'
$env:DAMA_WEIGHTS_FILE = 'weights\champion.pt'                # games measured, not assumed
launch.cmd preflight
```

With the engine it measures the compiled path, tries half precision and uses the
threads suggested by the core measurement; with a trained champion the length of the
games is measured instead of assumed, and the uncertainty narrows a lot.

### How long will it take

```bash
python benchmark.py --device cuda [--weights weights/champion.pt] [--engine engine_c/build/dama_engine]
```

It estimates the complete run: the cost of a cycle, its breakdown, the total and a
plausible interval, and writes a `bench_<date>.txt`. It is worth running before
committing days of compute: between configurations measured in this project there are
almost two orders of magnitude.

- It measures **every phase**, not only self-play: on the definitive runs the arena was
  about 59% of a cycle, self-play 30%, training 3%
  ([results §6](results.md#performance)). The generation ladder and the fixed-depth
  opponents are included with their own cadences.
- It goes through the **same mechanisms** as the run — the same worker pool, inference
  server or engine — because timing a game in a single process would leave out the cost
  of communication and the gain from batching.
- **The main source of error is game length**, which grows as the network learns
  (about 84 plies with a random network, 150-159 with a trained one). The benchmark
  measures the cost per ply and projects it onto the steady-state length; passing a
  trained champion with `--weights` makes that length measured.

Other measurements it can make:

| option | what it measures |
|---|---|
| `--cores-only` | the cores really usable: a container quota is read from the system, otherwise the machine is saturated and the work counted (also `python usable_cores.py`) |
| `--fp16-compare` | the same games with and without half precision (GPU only). A speed measurement, not a correctness one |
| `--sweep-arena --arena-threads 16 32 --arena-leaves 8 16 32 64` | the fastest arena configuration: each combination repeated, median and dispersion reported, a tie declared when intervals overlap |
| `--sweep` | the fastest worker/leaf/server-wait combination on the Python path |

On Windows `launch.cmd preflight -TuneArena` runs the arena tuning as part of the
preflight.

**Tune only on the machine that will run, and idle.** Repeating the same command on a
busy development machine reproduced the structural quantities exactly (game length,
batch sizes) and the times not at all — the same configuration measured 79.6 ms per ply
in one round and 7,227 in the next. **And do not raise the leaves for speed**: 16 leaves
cost 163 Elo against 8 ([results §6](results.md#leaf-batching-costs-playing-strength-and-a-lot-of-it)).
The threads, on the other hand, change only how many games run in parallel.

### A better estimate: project from the cycles already done

```bash
python tools/run_projection.py weights --cycles 100
```

It reads `metrics.csv` and projects from the real times, estimating the continuous
phases from the steady state and amortizing the periodic ones over their cadence. The
first cycles are structurally the cheapest (the replay window is still filling, games
are short, the periodic phases have not run yet), so multiplying the mean is wrong.
Replaying a real 71-cycle run that took 58.2 hours over 100 cycles:

| stopping at cycle | mean × 100 | `run_projection.py` |
|---|---|---|
| 10 | −28% | −28% |
| 15 | −21% | **−5%** |
| 20 | −11% | **+4%** |
| 30 | −7% | **+1%** |

Before cycle 20 the strength test has not run and its cost cannot be guessed, and the
tool says so. From cycle 20 the error stays below 5%, on the high side.

## 2. The configuration

The defaults **are** the final configuration: a production run needs no settings, and
the launchers choose device, engine and threads by themselves. Each value has a
measured reason, and knowing it avoids changing them on intuition
([results §7](results.md#7-final-configuration)).

| variable | default | why |
|---|---|---|
| `DAMA_C_PUCT` | 1.5 | measured by strength comparison |
| `DAMA_SIMS` | 400 | the search budget is not the constraint |
| `DAMA_GAMES` | 400 | games per cycle |
| `DAMA_BUFFER_SAMPLES` | 600000 | about ten cycles of window |
| `DAMA_VALUE_DISCOUNT` | 0.99 | measured on behavior |
| `DAMA_ARENA_TEMP_PLIES` | 10 | enough distinct arena games |
| `DAMA_CHANNELS` / `DAMA_BLOCKS` | 96 / 6 | 1.13 M parameters |
| `DAMA_EPOCHS` | 2 | beyond 10 the value head overfits |
| `DAMA_SPRT_MAX_GAMES` | 400 | cap of the promotion test |
| **`DAMA_MCTS_BATCH`** | **8** | **do not raise it: 16 costs 163 Elo, 32 costs 243** |
| `DAMA_CYCLES` | 100 | also sets the learning-rate schedule |

**`DAMA_CYCLES` cannot be changed after the start.** It governs the cosine decay of the
learning rate: a run started for 100 cycles and relaunched with 200 would see the
learning rate jump back up by a factor of 5.4 at cycle 101. Decide it before starting —
doubling the cycles bought +67 Elo on the definitive runs
([results §4.8](results.md#48-the-two-champions-face-to-face-doubling-the-cycles-paid-off)).
For a 200-cycle run in its own folder:

```powershell
$env:DAMA_WEIGHTS_DIR = 'weights_200'
$env:DAMA_CYCLES = '200'
launch.cmd
```

These change only speed, never the samples produced (verified bit for bit):

| variable | default | note |
|---|---|---|
| `DAMA_CPP_THREADS` | ¾ of the logical processors (`run.ps1`), all of them (`run.sh`) | from 13 to 24 threads the cost per ply dropped by 46% |
| `DAMA_DEVICE` | chosen by probing the card | |
| `DAMA_SPRT_CHUNK` | derived from the parallelism | not a free parameter |

`python conductor.py --help` lists every variable with its default. For a quick test of
the whole loop:

```bash
DAMA_CYCLES=2 DAMA_GAMES=4 DAMA_SIMS=25 DAMA_CHANNELS=32 DAMA_BLOCKS=2 python conductor.py
```

## 3. Launching

Do not start `conductor.py` by hand for a real run: use `./run.sh` on Linux and
`.\run.ps1` on Windows (or `launch.cmd`). The conductor starts with any configuration,
including a wrong one, and only writes it in the log; a run once went on the CPU with a
GPU sitting idle, and nobody noticed until the process died. The launchers check first,
and refuse to start if something is off:

- PyTorch imports, and the requested device is **really usable** — a real allocation,
  not `is_available()`. On Windows, with no device requested, a usable GPU is chosen
  automatically, and running on the CPU next to a usable GPU is a warning;
- the C++ engine, if enabled, exists, **can load a network** (an engine built without
  LibTorch cannot, and every cycle would silently fall back on Python) and **knows the
  options** the conductor passes (an engine older than the Python code would fall back
  too);
- the dry-run mode `DAMA_CPP_FAKE` is off: it produces samples with no playing value,
  and the loss goes down anyway;
- on Windows, the worker count on the Python path is within the safe limit (see
  [the known crash](#the-known-crash-on-windows));
- whether a previous state exists, in which case the run **resumes**.

Warnings stop the launch; `-Force` on Windows starts anyway.

```bash
./run.sh                               # Linux (DAMA_DEVICE defaults to cuda)
DAMA_CYCLES=200 ./run.sh
./watch.sh -f
```

```powershell
.\run.ps1                              # or launch.cmd
.\run.ps1 -Force
.\watch.ps1 -Follow                    # or launch.cmd watch -Follow
```

## 4. Following a run

The dashboard (`watch.sh`, `watch.ps1`) reads `metrics.csv`, `ladder.json` and the log,
and shows the latest cycles, strength against the fixed opponents, the generation
ladder, progress with the estimated end, and anomalies in the log (engine fallbacks,
exceptions, restarts). The pace is measured from the **timestamps** of the rows over the
last 20 intervals, the period of the strength tests, so the estimate does not swing
depending on whether one of those expensive cycles is in the window.

What a run writes in its weights folder (`DAMA_WEIGHTS_DIR`, default `weights`):

| file | content |
|---|---|
| `champion.pt` | the current champion |
| `train_state.pkl` | resume state: buffer and last completed cycle |
| `metrics.csv` | one row per cycle ([columns](design.md#9-measuring-progress)) |
| `ladder.json`, `gen_NNN.pt` | the generation ladder and its frozen generations |
| `run.log` | the log, when launched with `run.ps1` (`run.sh` writes `run.log` in the project folder unless `DAMA_LOG` says otherwise) |

## 5. Long unattended runs

**No need to guess how many cycles fit.** The state is rewritten atomically at every
cycle: the run can be stopped at any time, and relaunching resumes from the next cycle.
Launch the full number of cycles and stop when the time runs out — but remember that the
learning-rate schedule follows `DAMA_CYCLES`.

**Automatic restart after a crash** (Windows):

```powershell
launch.cmd -Retry 20
```

Every restart is printed and logged, where the dashboard counts it. If the conductor dies
within a minute the script does not retry: that is a wrong configuration, not the rare
crash. **To stop a run that restarts by itself**, create an empty file named `STOP` in
the weights folder, then press Ctrl-C:

```powershell
New-Item -ItemType File (Join-Path $env:DAMA_WEIGHTS_DIR 'STOP')
```

On Linux, run inside `tmux`, so the run survives the terminal.

**Keep the machine awake.** A computer that goes to sleep after half an hour without
keyboard activity stops the computation without any sign. On Windows:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

and postpone automatic restarts for system updates for the duration of the run.

**Disk space.** With a full window the resume checkpoint weighs about 2.4 GB and is
rewritten every cycle through a temporary file, so twice that is needed for a moment.
With the frozen generations, keep **at least 10 GB free**.

**Is it alive?** From another window, `launch.cmd watch -Follow` or `./watch.sh -f`: if
the cycle number does not advance between two readings, something has stopped.

## 6. Windows

### Starting on a machine that is not the development one

```
launch.cmd              checks and start of the run
launch.cmd watch        dashboard (add -Follow to refresh it by itself)
launch.cmd preflight    complete validation: build, tests, benchmark, resume
launch.cmd build        build the C++ engine
launch.cmd benchmark    how long the complete run would take here
```

**Use `launch.cmd` rather than the PowerShell scripts** when the project arrived as an
archive. Windows marks files extracted from a downloaded archive as coming from the
internet, and the default `RemoteSigned` policy refuses to run them; the message talks
about disabled scripts or missing signatures and sends one looking for the wrong fix —
changing the policy of the whole machine, which is a permanent security change and is
not needed. `launch.cmd` bypasses the check **only for the process it starts**. To use
the scripts directly, remove the mark once:

```powershell
Get-ChildItem -Recurse *.ps1 | Unblock-File
```

It needs Python with NumPy and PyTorch; a C++ compiler is needed only for the compiled
engine, and without it the pipeline still runs on the Python path. The first command to
give is `launch.cmd preflight`.

### The C++ engine on Windows

`engine_c\build.ps1` without LibTorch builds with g++, runs every test and produces an
engine with the **fake backend only**: enough to verify rules, search and concurrency,
not to play. The real engine needs Visual Studio with the C++ tools and LibTorch for
Windows:

```powershell
$env:LIBTORCH = 'C:\path\to\libtorch'
cd engine_c; .\build.ps1
```

This path is verified: built with MSVC, `parity_check` measured 0.000e+00 against
PyTorch, and the conductor ran complete cycles with the engine
([`engine_c/README.md`](../engine_c/README.md)). The script handles the three traps of
that build: the Visual Studio generator is multi-configuration and would build Debug
without `--config Release`; the executable is normalized into `build\`; and the LibTorch
DLLs are copied next to it, because without them a Windows executable dies before
`main` without printing anything. It also checks that the folder really is a LibTorch
for **Windows** (the Linux archive looks identical) whose version matches the Python
PyTorch. MinGW will not do: those libraries are built with MSVC.

### The known crash on Windows

On the development machine (Windows, 16 cores), parallel self-play on the **Python path**
with the default number of workers (cores − 1) and a full-size network can die of a
native `access violation` during training — not a Python exception, so the process
simply vanishes. The point changes every time, a sign of memory corruption rather than a
logic error, and with few games per cycle it sometimes does not happen. Tried without
success: limiting PyTorch threads, `KMP_DUPLICATE_LIB_OK=TRUE`. What works: a smaller
pool, **`DAMA_WORKERS=4`**.

The pool exists on the GPU too (the workers stay CPU processes next to the inference
server), so the risk does not go away with a graphics card. `run.ps1` warns about it
before starting. **It does not affect runs with the compiled engine**, which do not
create the pool at all ([results §6](results.md#portability)).

### Python-path parallelism on a GPU is not tuned for Windows

This concerns only the Python path. The defaults come from a Linux machine with dozens of
worker processes, where the inference server's queue is always full. On Windows the
four-worker cap means four workers sending eight leaves each: batches of about 32
positions against a limit of 256. The levers are `DAMA_WORKERS`, `DAMA_MCTS_BATCH` (which
costs strength, see above) and `DAMA_SERVER_WAIT` (seconds the server waits to merge
requests). They are found by measuring on the machine, with
`launch.cmd benchmark --sweep --device cuda`.

## 7. Linux GPU machines

### Is a rented machine usable?

```bash
bash check_pod.sh
```

A verdict in about fifteen seconds, before installing anything. `nvidia-smi` answering
does not mean CUDA works: it talks to the driver, while PyTorch needs the runtime. The
script asks the CUDA library for the **return code of `cuInit`**, which names the cause
(803: the library in the container does not match the host driver, not fixable from
inside; 802 or 999: worth one restart of the machine), and measures the cores really
available — declared cores are the host's, and the effective ones were between 25 and 43
on machines declaring many more.

### Preparing it

```bash
./setup_pod.sh
```

In order, stopping at the first failure: whether CUDA initializes and what the CPU quota
is; tools and GPU capability; the PyTorch build for that architecture, gated by running a
real convolution on the card; LibTorch in the exact version of PyTorch; the C++ engine
with CMake and LibTorch; the parity check between the engine and PyTorch; and the
preflight. It ends with the next steps: optionally calibrate `c_puct`, then start the run
inside tmux with `./run.sh`, and follow it with `./watch.sh -f`.

## 8. The C++ engine

Build and validation are described in [`engine_c/README.md`](../engine_c/README.md). The
conductor delegates self-play and arena to it when `DAMA_CPP_ENGINE` points to the
executable (the launchers set it when the built engine exists); training stays in
PyTorch.

```bash
DAMA_DEVICE=cuda DAMA_CPP_ENGINE=./engine_c/build/dama_engine python conductor.py
```

| variable | default | |
|---|---|---|
| `DAMA_CPP_ENGINE` | empty (Python path) | path of the executable |
| `DAMA_CPP_THREADS` | number of cores | engine threads |
| `DAMA_CPP_FP16` | 0 | half-precision inference (CUDA only) |
| `DAMA_CPP_FAKE` | 0 | run the engine without a network, to test the plumbing only |

Before producing results with a newly built engine, check it against PyTorch on the
machine that has LibTorch — it must print `PARITY OK`:

```bash
python engine_c/tools/export_jit.py weights/champion.pt model_jit.pt cuda
./engine_c/build/parity_check model_jit.pt cuda
```

To exercise the connection without LibTorch (the samples have no playing value):

```bash
DAMA_CPP_ENGINE=./engine_c/dama_engine DAMA_CPP_FAKE=1 DAMA_CYCLES=2 DAMA_GAMES=4 python conductor.py
```

## 9. Playing and measuring

```bash
python play.py --weights results/learning_curve              # you play White
python play.py --weights results/learning_curve --side black --sims 100
python play.py --weights results/learning_curve --vs-ab 4 --games 20
```

During a game, type the number of the move in the list, or `h` for a hint, `u` to take
back a move, `s N` to change the engine's simulations, `q` to quit. Dark squares are
numbered 1 to 32. Moves are chosen by their row in the list because, with capture
priority, two different chains can start and end on the same squares.

```bash
DAMA_WEIGHTS_DIR=cal DAMA_CPP_ENGINE=./engine_c/build/dama_engine DAMA_DEVICE=cuda python calibrate.py
```

`calibrate.py` compares search settings by making one network play itself with two
different values, including a null check of the reference against itself; its
`DAMA_CAL_*` variables are listed by `--help`.

**Promotion safeguard** (off by default, see [design §7](design.md#7-promotion)):

```powershell
$env:DAMA_ANTIDRIFT = '40'        # games against a frozen generation
$env:DAMA_ANTIDRIFT_BACK = '3'    # how many ladder generations back (about 30 cycles)
$env:DAMA_ANTIDRIFT_MIN = '0.5'   # below this score the promotion is refused
```

### Analysis tools

The tools behind [`results.md`](results.md), in `tools/`. Each answers `--help`.

| tool | what it answers |
|---|---|
| `kingsrow.py` | matches against Kingsrow Italian through CheckerBoard (`--validate` checks the board mapping, `--save` records the games) |
| `rules_crosscheck.py` | five positions to check that another engine applies the same capture priority |
| `read_games.py` | a recorded game file, sorted by length |
| `quiet_material.py` | material balance in quiet positions; `--csv` regenerates the figure data |
| `anchors.py` | absolute strength of every saved generation, without training |
| `compare.py` | two trained networks against each other, with a confidence interval |
| `sweep_train.py` | training parameters compared on the same frozen buffer |
| `diagnose_targets.py` | how much the network still has to learn from its own targets, and value calibration |
| `target_noise.py` | how much of the policy target is unpredictable noise |
| `ab_value_discount.py` | paired A/B test of the value discount |
| `draw_rule_check.py` | how far the implemented draw rule diverges from the official one |
| `run_projection.py` | duration of the run, projected from the cycles already done |

`export_onnx.py` is not an analysis tool: it converts a checkpoint into ONNX, the format
that runs a network outside PyTorch — in a browser through onnxruntime-web, for instance.
It checks the conversion the way the C++ backend is checked: it runs the exported file and
PyTorch on the same real positions and compares logits, value, the priors the search would
consume and the move that would be played, refusing a file that would play differently.
The architecture is read from the checkpoint, so nothing has to be remembered about how it
was trained. It needs `onnx` and `onnxruntime`, which training and playing do not.

```bash
pip install onnx onnxruntime
python tools/export_onnx.py results/learning_curve champion.onnx
```

### The browser engine

`web/` holds the rules, the encoder and the search ported to JavaScript, so a page can
play against a published network with no server behind it. It has no build step — plain ES
modules — and its tests are held to the same references as the C++ engine: the perft
counts, the encoder checksums, and the visit counts of `mcts.py` itself.

```bash
node --test web/test/*.test.js
```

The network runs there too, through ONNX Runtime Web: `web/parity.html` puts the whole
browser stack — ported encoder, exported model, WebAssembly runtime — against PyTorch on the
same positions, and it agrees to about 1e-6 on the value and on every prior.

```bash
python -m http.server 8765 --directory web    # a model cannot be fetched from file://
```

`web/index.html` is the board itself: pick one of the seven published networks, pick how
long it may think, and play. The search runs in a worker, so the page keeps drawing while
the engine works, and a finished game can be saved in the format of `results/games/`.

See [`web/README.md`](../web/README.md).

## 10. What is verified

The Python code has no operating-system dependency other than the lock on the weights
folder, and the C++ code uses no POSIX headers or compiler extensions. On Windows the whole
test suite and the preflight pass, `run.ps1` completes cycles and `watch.ps1` reads the
metrics of a real run; the definitive runs A and B ran on Windows with a graphics card and
the compiled engine. On Linux the whole chain, GPU included, ran on rented machines.
`setup_pod.sh` and `check_pod.sh` exist only for Linux, where rented machines are.

The browser port is held to the same references as the C++ engine — the perft counts, the
encoder checksums and the visit counts of `mcts.py` — and `web/parity.html` compares the
whole page, network included, against PyTorch on real positions.

## 11. Environment variables

Every Python entry point that is configured through the environment lists its own
variables, with their defaults, when run with `--help`: `conductor.py`,
`calibrate.py`, and the tools in `tools/`. The launch scripts read a few more:

| variable | read by | default | meaning |
|---|---|---|---|
| `BENCH_GAMES` | `preflight.ps1`, `preflight.sh` | 8 | self-play games of the benchmark step |
| `BENCH_ARENA_GAMES` | `preflight.ps1`, `preflight.sh` | 30 | arena games of the benchmark step |
| `BENCH_SIMS` | `preflight.ps1`, `preflight.sh` | 400 | simulations per move in the benchmark step |
| `BENCH_WORKERS` | `preflight.ps1`, `preflight.sh` | 4 | worker processes when the benchmark runs the Python path |
| `DAMA_WEIGHTS_FILE` | `preflight.ps1`, `preflight.sh` | none | trained champion for the benchmark, so game length is measured |
| `DAMA_LOG` | `run.*`, `watch.*` | `run.log`: in the weights folder for `run.ps1` and `watch.ps1`, in the current folder for `run.sh` and `watch.sh` | log file |
| `DAMA_PACE_WINDOW` | `watch.*` | 20 | cycle intervals averaged for the pace |
| `DAMA_WATCH_EVERY` | `watch.*` | 60 | seconds between refreshes in follow mode |
| `LIBTORCH` or `DAMA_LIBTORCH` | `engine_c/build.ps1`, `preflight.ps1` | none | LibTorch folder; without it the engine gets the fake backend |
| `DAMA_LIBTORCH` | `setup_pod.sh` | `/workspace/libtorch` | where LibTorch is installed on a rented machine |
| `DAMA_CUDA_TAG` | `setup_pod.sh` | deduced from the GPU (`cu128` for capability 10 and above, `cu124` otherwise) | PyTorch build to install |
| `DAMA_SKIP_PREFLIGHT` | `setup_pod.sh` | 0 | `1` skips the final preflight |
