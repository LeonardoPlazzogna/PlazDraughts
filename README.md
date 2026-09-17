# PlazDraughts

[![selfchecks](https://github.com/LeonardoPlazzogna/PlazDraughts/actions/workflows/selfchecks.yml/badge.svg)](https://github.com/LeonardoPlazzogna/PlazDraughts/actions/workflows/selfchecks.yml)

A self-play engine for **Italian draughts** (*dama italiana*) in the AlphaZero
style, trained from zero game knowledge and measured against fixed-depth
opponents, its own frozen generations and an external engine. It is the
Italian-draughts companion of
[PlazChess](https://github.com/LeonardoPlazzogna/PlazChess), and the two are
studied in the same Bachelor's thesis.

It can also be **[played in a browser](https://leonardoplazzogna.github.io/PlazDraughts/)**:
`web/` holds the same rules, encoder and search in JavaScript, running the
trained networks through ONNX Runtime, and it is checked against the Python
engine rather than merely resembling it. The `pages` workflow publishes that
folder on every push; locally, `python -m http.server --directory web` and open
`index.html`.

The project combines a **Python** training pipeline — a PyTorch residual
policy-value network with Squeeze-and-Excitation blocks and a win/draw/loss
value head, Monte Carlo Tree Search, a windowed replay buffer — with a **C++20
engine** that plays the self-play and arena games: move generation, batched
multi-threaded MCTS and LibTorch inference. A conductor repeats the
**SELF-PLAY → TRAIN → ARENA** cycle; a sequential probability ratio test
(SPRT) decides whether the trained candidate replaces the champion.

Everything is implemented from scratch: the rules and move generation
(including the capture-priority hierarchy of the Italian game), the encoding,
the search, the training loop, the engine and the measurement tools around them.
PyTorch and LibTorch provide only the tensor and inference layer; no draughts
library and no reinforcement-learning framework is used.

## Results

Unlike PlazChess, this project was trained to completion. Two definitive runs,
with every measurement fix of [§5](docs/results.md#5-measurement-defects-found-and-fixed)
in place, on a single machine with a graphics card:

| | Run A | Run B |
|---|---|---|
| Cycles (self-play games) | 100 (40,000) | 200 (80,000) |
| Duration | 31.0 h | 68.2 h |
| Against weakened Kingsrow Italian, 50 games | 0 won, 27 drawn, 23 lost | 0 won, 30 drawn, 20 lost |
| Against its own frozen generations | Detectable progress until about cycle 40–50 | Detectable until about cycle 160–180 |

What the measurements say, in short:

- **The network learns, and quickly exhausts the fixed opponents.** Within 40–60
  cycles it wins practically every game against alpha-beta at depths 4, 6 and 8 —
  which also means those opponents stop measuring progress
  ([§4.3](docs/results.md#43-against-fixed-depth-opponents)).
- **It does not beat a real engine.** Against Kingsrow Italian, deliberately
  weakened to 0.01 s per move with no opening book, it scores only draws, all of
  them closed by the no-progress rule, and replaying the games shows that it never
  builds a lasting material advantage
  ([§4.1](docs/results.md#41-against-an-external-engine),
  [§4.5](docs/results.md#45-the-mechanism-of-the-endgame-defect)). That rule is a
  simplification of the official one and ends games earlier than the federation
  would ([§9](docs/results.md#9-the-implemented-draw-rule-is-not-the-official-one)).
- **Doubling the cycles bought +67 Elo** [+18, +119] in the direct match between
  the two champions. Fixing the measurement instruments had bought +164 over an
  earlier 300-cycle run, and a single badly tuned performance setting — the leaf
  batch size of the search — costs up to −293
  ([§4.8](docs/results.md#48-the-two-champions-face-to-face-doubling-the-cycles-paid-off),
  [§6](docs/results.md#performance)).
- **Relative measurements can contradict absolute ones.** A promotion accepted by
  the arena and celebrated by the generation ladder made the champion clearly
  weaker against a fixed opponent
  ([§4.6](docs/results.md#46-when-the-three-measurements-contradict-each-other-the-case-of-cycle-60)).

Every number, with its conditions and the tool that produced it, is in
[`docs/results.md`](docs/results.md), including the measurements that were
withdrawn and why. The data series behind the thesis figures, the recorded games
and a trained champion are in [`results/`](results/).

## Repository layout

```
dama/                 rules: board, move generation (mandatory capture, capture
                      priority, promotion, men cannot capture kings, non-flying
                      kings), game state, encoder
model.py              policy-value network: residual trunk with SE blocks,
                      256-action policy head, win/draw/loss value head
mcts.py               PUCT search with leaf batching and virtual loss
evaluators.py         network evaluator and uniform evaluator
selfplay.py           one game played by MCTS -> training samples
train.py              training on the replay buffer
conductor.py          the SELF-PLAY -> TRAIN -> ARENA loop, resumable
parallel.py           persistent pool of self-play workers (Python path)
inference_server.py   one process owns the GPU and batches the workers' requests
cpp_engine.py         bridge to the C++ engine; cpp_dataset.py reads its datasets
sprt.py               sequential test for the promotion decision
ladder.py             Elo ladder of frozen generations
players.py            fixed opponents: random, greedy, fixed-depth alpha-beta
metrics.py            one CSV row per cycle
runlock.py            exclusive lock on the weights folder
play.py               play against the champion, or measure it against an opponent
calibrate.py          calibrate search parameters by direct strength comparison
benchmark.py          estimate how long a complete run takes on this machine
usable_cores.py       how many cores can really be used
game_diversity.py     strength and number of distinct games against alpha-beta
engine_c/             the C++20 engine (see engine_c/README.md)
web/                  the same engine in a browser, and the page you play on
tools/                analysis tools behind docs/results.md
tests/                test suite (plain scripts, no framework needed)
results/              figure data, recorded games, a trained champion
docs/                 results.md, running.md, design.md
run.ps1, run.sh       launchers that check the configuration before starting
watch.ps1, watch.sh   dashboard of a run in progress
preflight.ps1/.sh     build, tests, benchmark and resume test in one command
launch.cmd            Windows entry point
setup_pod.sh          prepares a rented Linux GPU machine; check_pod.sh vets one
```

## Requirements

- **Python 3.11 or later** with NumPy and PyTorch. PyTorch is installed separately,
  because the right build depends on the machine: see
  [`requirements.txt`](requirements.txt).
- For the C++ engine: a **C++20 compiler** (g++ or clang on Linux, g++ or MSVC on
  Windows). The neural backend also needs **CMake** and **LibTorch** matching the
  installed PyTorch; without them the engine builds with a fake backend that is
  enough for all the tests.
- A **CUDA GPU** is needed to train in reasonable time, not to run the code:
  everything also runs on the CPU, about two orders of magnitude slower
  ([§6](docs/results.md#performance)).
- Optional: Kingsrow Italian inside CheckerBoard (64-bit Windows) for
  `tools/kingsrow.py`.

## Quick start

```bash
pip install -r requirements.txt          # then PyTorch, as the file explains
for t in tests/test_*.py; do python "$t" || break; done
```

Play against the trained champion included in `results/learning_curve/`, or
measure it against fixed-depth alpha-beta:

```bash
python play.py --weights results/learning_curve
python play.py --weights results/learning_curve --vs-ab 4 --games 20
```

Build and validate the C++ engine (perft, encoder parity, MCTS, multi-thread hub,
self-play):

```bash
cd engine_c && ./build.sh                # .\build.ps1 on Windows
```

Start a training run. The launchers check the configuration first — device,
engine, workers, previous state — and refuse to start on a wrong one:

```bash
./run.sh                                 # Linux
./watch.sh -f                            # dashboard, refreshes every 60 s
```

```powershell
launch.cmd preflight                     # Windows: build, tests, benchmark, resume test
launch.cmd                               # start the run
launch.cmd watch -Follow
```

The defaults are the final configuration
([§7](docs/results.md#7-final-configuration)). Every parameter can be overridden
through a `DAMA_*` environment variable: `python conductor.py --help` lists them
with their defaults. Running, monitoring, rented machines, benchmarking and long
unattended runs are covered in [`docs/running.md`](docs/running.md); the design
of the pipeline and the reasons behind it in [`docs/design.md`](docs/design.md).

## Validation

These checks need neither a GPU nor LibTorch:

| Check | What it verifies |
|---|---|
| `tests/test_rules.py` | The rules: mandatory capture, capture priority, promotion, men not capturing kings, non-flying kings, terminal positions, encoder and action round-trip |
| `tests/test_mcts_rules.py` | Search behavior: sign of the backed-up value, virtual-loss bookkeeping, terminal values, no-progress draw, determinism |
| `tests/test_alphazero.py` | MCTS, self-play and training end to end |
| `tests/test_metrics.py` | Per-cycle metrics, monotonicity of the alpha-beta ladder, transposition-table equivalence |
| `tests/test_sprt.py` | The SPRT's real error rates, by Monte Carlo simulation |
| `tests/test_ladder.py` | Elo arithmetic and persistence of the generation ladder |
| `tests/test_inference_server.py` | Parity between the inference server and the local evaluator |
| `tests/test_runlock.py` | The weights-folder lock, including release after its holder is killed |
| `tests/test_cpp_dataset.py` | Datasets written by the C++ engine are usable by the Python training |
| `tests/test_cpp_conductor.py` | The Python–C++ bridge, arena scoring and the fallback when the engine fails |
| `engine_c/build.sh`, `build.ps1` | Perft to depth 7 against the Python reference, encoder parity, MCTS invariants, concurrency, self-play |
| `web/test/*.test.js` | The browser port: the same perft counts, the same encoder checksums, the visit counts of `mcts.py` itself, and the game format `tools/read_games.py` reads |

All of them run in CI on every push — the C++ engine, the Python suite on 3.11 and
3.14 with a check that every entry point answers `--help`, and the browser tests on
Node; the badge at the top of this page reports their current state.

With LibTorch, `engine_c/parity_check` compares the C++ backend with PyTorch on the
same traced model (maximum difference 0.000e+00 measured,
[§6](docs/results.md#6-implementation-correctness-checks)). `preflight.ps1` and
`preflight.sh` chain the build, the tests, the core measurement, the benchmark and
a resume test.

## Third-party software

- **Kingsrow Italian**, by Ed Gilbert, is used only as an external opponent,
  through the CheckerBoard engine interface (`tools/kingsrow.py`). It is not
  included in this repository and must be installed separately. The recorded games
  in `results/games/` and `results/learning_curve/` contain its moves.
- PyTorch, LibTorch and NumPy are installed separately. No third-party code is
  vendored in this repository.

## Citation

The accompanying Bachelor's thesis — *The Algorithmic Emergence of Strategic
Reasoning: An End-to-End Deep Reinforcement Learning Approach to Chess and Italian
Draughts* (Politecnico di Milano, Mathematical Engineering, 2026) — studies PlazChess
as its primary implementation and this repository as the port of the same method
to Italian draughts. Citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT](LICENSE) license.
