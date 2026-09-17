# `web/` — the engine that runs in a browser

The rules, the encoder and the search of this project, ported to JavaScript so a
page can play against the trained networks with no server behind it. The network
itself is the same file the Python code trains, converted to ONNX by
[`tools/export_onnx.py`](../tools/export_onnx.py) and run by onnxruntime-web.

**Status: it plays.** `index.html` is a board you can play on, against any of
the seven published networks, with the search running in a worker so the page
stays responsive. `parity.html` is the proof that it is the same engine: the
same positions through PyTorch and through this page agree to about 1e-6 on the
value and on every prior, and never prefer a different move.

## Why a port and not the C++ engine compiled to WebAssembly

`engine_c/` is already validated, and Emscripten can compile it. Two things get
in the way. Its search is synchronous and batch-oriented while the browser's
inference is asynchronous, so the two would have to be bridged with Asyncify;
and its threads want `SharedArrayBuffer`, which needs COOP/COEP headers that
GitHub Pages does not let you set. The port is 439 lines of code — board, moves,
game, encoder, search — and it can be held to the same references the C++ is held
to, which is what the tests below do.

## No build step, on purpose

Plain ES modules: the browser loads them as they are, and `node --test` runs the
tests with nothing installed. Anyone who clones the repository can check the
engine in a second, without a bundler and without a lockfile.

`package.json` is here for one line only — `"type": "module"` — because without
it Node reads a `.js` file as CommonJS and the imports fail on anything but the
newest releases. It declares no dependencies, so `npm install` has nothing to do.

```bash
node --test web/test/*.test.js
```

## What is checked, and against what

Nothing here is checked against itself. Every expected number comes from the
implementations that already agree with each other:

| test | oracle |
|---|---|
| `rules.test.js` | the perft counts of `engine_c/perft.cpp` (1, 7, 49, 302, 1469, 7361, 36473, 177532) and the five positions of `tools/rules_crosscheck.py`, one per Italian capture-priority rule |
| `encoder.test.js` | the plane checksums of `engine_c/encoder_parity.cpp`, on hand-built positions and along a deterministic playout |
| `search_parity.test.js` | the visit counts of `mcts.py` itself, through `test/fixtures/search.json` |
| `search.test.js` | behaviour with a uniform evaluator: it must find a move that ends the game, refuse one that loses, and obey a forced capture |

The search fixture deserves a word. With a uniform evaluator and no Dirichlet
noise, nothing in the search is random: the tree is a deterministic function of
the rules, the PUCT formula, the leaf batching and the virtual loss. The two
implementations must therefore agree on every visit count, and they do. Regenerate
the fixture only when the search changes on purpose:

```bash
python web/test/fixtures/make_fixtures.py
```

`parity.html` has a reference of its own, `test/fixtures/net.json`: what PyTorch
answers on the same positions, written by `make_net_fixture.py`. Regenerate it
when the published network changes:

```bash
python web/test/fixtures/make_net_fixture.py results/learning_curve
```

## The files

```
engine/board.js       squares, pieces, the 32 dark squares
engine/moves.js       move generation and the capture-priority rules
engine/game.js        Position, transitions, outcome, draw rule
engine/encoder.js     the 7 planes, the 256-slot action space, the priors
engine/evaluators.js  what the search asks: uniform, or a network
engine/mcts.js        flat-tree PUCT search with leaf batching
engine/net.js         the ONNX model, run by ONNX Runtime in WebAssembly
app/worker.js         the engine, kept off the page's thread
app/ui.js             the board, the clicks, the conversation with the worker
app/notation.js       squares 1..32 and the game text of results/games/
index.html            the page you play on
networks.html         what each published network is, and what it measured
parity.html           the end-to-end check against PyTorch
models/               the seven published networks and their manifest
vendor/               ONNX Runtime Web, copied in (see vendor/README.md)
```

## Playing

A move is a PATH, not a pair of squares: two capture chains can start with the
same jump, so a click extends a prefix and the move is played the moment the
prefix is a whole legal move. The board shows which pieces can move, where the
current chain can continue, what the last move touched and what it captured.

The opponent is any of the published networks, from the untrained one to the
300-cycle champion, and the thinking budget goes from 50 to 800 simulations.
A finished game can be saved as a text file in the format of
[`results/games/`](../results/games), which `tools/read_games.py` and
`tools/quiet_material.py` already read.

## What it costs to play

A move costs about the number of simulations times the cost of one position, and
batching barely changes the second factor. On a desktop browser, single-threaded,
that came to a few milliseconds per position — roughly a second for a move at 400
simulations. Do not take the figure on trust: `parity.html` prints the cost per
position measured on the machine you open it with. The first call also loads
14 MB of runtime and 4.5 MB of weights, which the browser then caches.

The check itself needs a server, because a page opened from the filesystem
cannot fetch a model:

```bash
python -m http.server 8765 --directory web
# then open http://127.0.0.1:8765/parity.html
```

## Publishing it

`.github/workflows/pages.yml` puts this folder online, at
<https://leonardoplazzogna.github.io/PlazDraughts/>, on every push to `main`,
after re-running the tests and checking that every network still matches its
checksum in the manifest. Nothing is built and nothing is downloaded at deploy
time: what goes online is what a clone serves.

Pages has to be switched on once by hand, in Settings → Pages → Source →
“GitHub Actions”. Until that is done the deploy step fails and nothing else is
affected.
