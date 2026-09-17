"""
Generates the fixture that holds the BROWSER NETWORK to the PyTorch one.

    python web/test/fixtures/make_net_fixture.py results/learning_curve

The engine tests already prove that the browser generates the same moves and
encodes the same planes. What they cannot see is the last step: the planes
entering an ONNX model in WebAssembly and coming back as priors. A wrong plane
order, a model exported from other weights, a runtime that quietly disagrees --
none of it crashes, it only plays worse.

So this writes, for a handful of REAL positions, what PyTorch answers: the
value and the prior of every legal move. web/parity.html runs the same
positions through the page's own stack and compares. The tolerance is not
arbitrary: tools/export_onnx.py measures the ONNX-against-PyTorch difference at
about 1e-5 on the logits, so anything at 1e-3 on the priors is a different
computation, not arithmetic.

The default model is the published champion, the only weights in the repository.
"""
from __future__ import annotations
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from dama import Position                                  # noqa: E402
from evaluators import NetEvaluator                        # noqa: E402
from tools.export_onnx import load_network, sample_positions   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.json")
N_POSITIONS = 24
SEED = 20260916


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else "results/learning_curve"
    net, arch, src = load_network(weights)
    ev = NetEvaluator(net, device="cpu")

    positions = sample_positions(N_POSITIONS, SEED)
    cases = []
    for pos in positions:
        priors, value = ev.evaluate(pos)
        cases.append({
            "board": list(pos.board),
            "turn": pos.turn,
            "no_progress": pos.no_progress,
            "value": round(float(value), 6),
            "priors": sorted(([m.frm, m.to, round(float(p), 6)] for m, p in priors.items()),
                             key=lambda t: (t[0], t[1])),
        })

    out = {
        "source": os.path.basename(src),
        "architecture": {k: arch[k] for k in ("channels", "n_blocks", "in_planes")},
        "positions": cases,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, separators=(",", ":")))
        fh.write("\n")
    print(f"  {len(cases)} positions from {src}")
    print(f"  written {OUT} ({os.path.getsize(OUT) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
