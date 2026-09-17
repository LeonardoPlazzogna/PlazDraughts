"""
Exports a trained network to ONNX, the format a browser can run.

    python tools/export_onnx.py weights/champion.pt champion.onnx
    python tools/export_onnx.py weights champion.onnx --fp16
    python tools/export_onnx.py results/learning_curve champion.onnx --positions 200

Why a second export format. `engine_c/tools/export_jit.py` produces TorchScript
for the C++ engine, which links LibTorch; a browser has neither, and runs the
network through onnxruntime-web instead. Same weights, different container.

THE ARCHITECTURE IS READ FROM THE CHECKPOINT, not declared: channels, blocks,
input planes, the width of the value head and whether the squeeze-and-excitation
blocks are there all follow from the shapes in the state dict. A published
checkpoint therefore exports correctly without anyone having to remember with
which environment variables it was trained -- and a mismatch surfaces as a load
error here instead of as a network that plays badly later.

THE PARITY CHECK IS THE POINT OF THE TOOL, exactly as for the C++ backend: the
risk of a second implementation is not slowness, it is a network that plays in a
subtly different way without anyone noticing. So the exported file is run
through onnxruntime on REAL positions -- reached by legal play, not random
tensors -- and compared with PyTorch on three levels:

    logits and value   the raw output, where any difference starts
    priors             what the search actually consumes, after masking the
                       illegal moves and the softmax
    chosen move        the only difference a player would ever see

It exits non-zero if the tolerance is exceeded, so it can be used as a check and
not only as a converter.

Requires `onnx` and `onnxruntime` (not needed to train or to play, hence not in
requirements.txt):

    pip install onnx onnxruntime
"""
from __future__ import annotations
import argparse
import copy
import os
import random
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import Position, IN_PLANES, POLICY_SIZE       # noqa: E402
from dama.encoder import encode                          # noqa: E402
from evaluators import priors_from_logits                # noqa: E402
from model import PolicyValueNet                         # noqa: E402

# Opset 17 is the highest that onnxruntime-web supports across its backends. The
# network needs nothing exotic -- convolutions, batch norm, a couple of matmuls
# -- so a newer opset would buy nothing and cut out browsers.
DEFAULT_OPSET = 17


def architecture_from_state_dict(sd) -> dict:
    """The shape of the network, deduced from the weights themselves."""
    stem = sd["stem.0.weight"]
    blocks = {k.split(".")[1] for k in sd if k.startswith("trunk.")}
    return dict(
        in_planes=int(stem.shape[1]),
        channels=int(stem.shape[0]),
        n_blocks=len(blocks),
        use_se=any(k.startswith("trunk.0.se.") for k in sd),
        v_channels=int(sd["v_conv.0.weight"].shape[0]),
        v_hidden=int(sd["v_fc1.weight"].shape[0]),
    )


def load_network(path: str):
    """Loads champion.pt from a file or from a folder that contains it."""
    if os.path.isdir(path):
        path = os.path.join(path, "champion.pt")
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Give the checkpoint or the folder that holds it.")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    arch = architecture_from_state_dict(sd)
    net = PolicyValueNet(**arch)
    net.load_state_dict(sd, strict=True)    # strict: a silent mismatch is worse
    net.eval()
    return net, arch, path


def sample_positions(n: int, seed: int) -> list[Position]:
    """Positions reached by legal play, from the opening to the endgame.

    Random tensors would exercise the arithmetic but not the network's working
    range: the planes of a real position are almost all zeros and obey the rules
    of the game. A difference that only shows up on real inputs is precisely the
    one a parity check on noise would miss.
    """
    rng = random.Random(seed)
    out, pos = [], Position()
    while len(out) < n:
        moves = pos.legal_moves()
        if not moves or pos.is_terminal():
            pos = Position()
            continue
        out.append(pos)
        pos = pos.play(rng.choice(moves))
    return out


def main():
    ap = argparse.ArgumentParser(description="export a network to ONNX and check parity")
    ap.add_argument("weights", help="champion.pt, or the folder that holds it")
    ap.add_argument("out", help="path of the .onnx file to write")
    ap.add_argument("--fp16", action="store_true",
                    help="half precision: half the file, looser parity")
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    ap.add_argument("--positions", type=int, default=128,
                    help="real positions used for the parity check")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--tolerance", type=float, default=None,
                    help="max |diff| allowed on logits and value "
                         "(default 1e-4 in fp32, 5e-2 in fp16)")
    args = ap.parse_args()

    # Asked for BEFORE reading the weights: torch.onnx.export needs the `onnx`
    # package to serialize, and without it fails deep inside the exporter with a
    # traceback that says nothing about what to install.
    try:
        import onnx           # noqa: F401
    except ImportError:
        sys.exit("onnx is not installed, and torch.onnx.export needs it to write the "
                 "file.\n  pip install onnx onnxruntime")

    net, arch, src = load_network(args.weights)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"loaded {src}")
    print(f"  network {arch['channels']}x{arch['n_blocks']}, {arch['in_planes']} planes, "
          f"SE {'on' if arch['use_se'] else 'off'}, {n_params/1e6:.2f} M parameters")

    # WHY THE OLD EXPORTER (dynamo=False). The torch.export-based one, the
    # default since PyTorch 2.9, writes the weights to a SEPARATE .onnx.data
    # file: two files to fetch and to keep in step, where a page wants one. It
    # also refuses to emit the requested opset, silently moving up. The old
    # path is deprecated and will eventually go: when it does, the sidecar file
    # has to be shipped next to the model and the loader told about it.
    #
    # The two warnings below are expected and covered by the parity check: the
    # deprecation, and a note that advanced indexing (the gather of the 32 dark
    # squares) is rebuilt from several ONNX operators and would break on
    # NEGATIVE indices -- these are constant buffers built from square numbers,
    # so they never are.
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            module="torch.onnx")
    warnings.filterwarnings("ignore", message=".*advanced indexing.*")

    # The batch axis stays dynamic: the browser evaluates one position while the
    # user thinks, and batches of leaves during the search. A fixed batch would
    # force a separate file per batch size.
    example = torch.zeros(4, arch["in_planes"], 8, 8)
    # A COPY is converted, never `net` itself: .half() modifies the module in
    # place, and .float() afterwards does not bring the lost bits back. The
    # reference would then be a network already rounded to half precision, and
    # the check would compare fp16 against fp16 while claiming to measure the
    # distance from fp32 -- the most flattering possible mistake.
    export_net = copy.deepcopy(net).half() if args.fp16 else net
    if args.fp16:
        example = example.half()

    torch.onnx.export(
        export_net, example, args.out,
        input_names=["planes"],
        output_names=["policy", "value", "wdl"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"},
                      "value": {0: "batch"}, "wdl": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"written {args.out}  ({size_mb:.1f} MB, opset {args.opset}, "
          f"{'fp16' if args.fp16 else 'fp32'})")

    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime is not installed: the file was written but NOT checked.")
        print("  pip install onnxruntime      then run this tool again")
        return 0

    positions = sample_positions(args.positions, args.seed)
    planes = np.stack([encode(p) for p in positions]).astype(np.float32)

    with torch.no_grad():
        t_logits, t_value, _ = net(torch.from_numpy(planes))
    t_logits = t_logits.numpy()
    t_value = t_value.numpy().reshape(-1)

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    feed = {"planes": planes.astype(np.float16 if args.fp16 else np.float32)}
    o_logits, o_value, _ = sess.run(None, feed)
    o_logits = o_logits.astype(np.float32)
    o_value = o_value.astype(np.float32).reshape(-1)

    d_logits = float(np.max(np.abs(t_logits - o_logits)))
    d_value = float(np.max(np.abs(t_value - o_value)))

    # What the search consumes is not the logits but the priors: illegal moves
    # masked out, then a softmax over what is left. Two sets of logits that
    # differ by a hair can still give the same priors -- and it is the priors,
    # with the value, that decide how the engine plays.
    d_priors, changed, margins = 0.0, 0, []
    for i, pos in enumerate(positions):
        pt = priors_from_logits(pos, t_logits[i])
        on = priors_from_logits(pos, o_logits[i])
        if not pt:
            continue
        d_priors = max(d_priors, max(abs(pt[m] - on.get(m, 0.0)) for m in pt))
        if max(pt, key=pt.get) != max(on, key=on.get):
            changed += 1
            # How close the two moves were IN PYTORCH. A move that changes
            # where the reference itself had a near-tie says the export lost
            # precision on a coin flip; one that changes where the reference
            # was decided says the export computes something else.
            top2 = sorted(pt.values(), reverse=True)[:2]
            margins.append(top2[0] - top2[1] if len(top2) > 1 else top2[0])

    tol = args.tolerance if args.tolerance is not None else (5e-2 if args.fp16 else 1e-4)
    print(f"\nparity against PyTorch on {len(positions)} real positions:")
    print(f"  logits     max |diff| = {d_logits:.3e}")
    print(f"  value      max |diff| = {d_value:.3e}")
    print(f"  priors     max |diff| = {d_priors:.3e}")
    print(f"  best move  differs in {changed} of {len(positions)} positions")
    if margins:
        print(f"             where PyTorch itself was within "
              f"{min(margins):.1e}..{max(margins):.1e} of a tie")

    if d_logits > tol or d_value > tol:
        print(f"\nFAILED: above the tolerance of {tol:.0e}. The exported file does NOT "
              "compute what PyTorch computes.")
        return 1
    if changed:
        print("\nFAILED: the two implementations choose a different move somewhere. "
              "Whatever the numbers say, they do not play the same game.")
        return 1
    print(f"\nOK: within {tol:.0e}, and the chosen move is the same everywhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
