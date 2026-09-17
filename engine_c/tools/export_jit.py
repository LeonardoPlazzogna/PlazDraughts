"""
Exports the Python network to TorchScript, in the format the C++ engine loads
(engine_c/libtorch_backend.hpp), and generates the data for the parity check.

Usage:
    python engine_c/tools/export_jit.py weights/champion.pt model_jit.pt [cpu|cuda]

The third argument is the device on which the REFERENCE is computed (default
cpu). Choose the same one parity_check will run on: CPU and GPU give slightly
different results (different kernels, and on Ampere+ GPUs also TF32, 10 bits of
mantissa instead of 23, when it is enabled), so comparing a CUDA backend against
a CPU reference produces differences of ~1e-3 that are NOT errors.

Produces:
    model_jit.pt        the traced model, to pass to --weights
    model_jit.pt.input  test input [B,7,8,8], raw float32
    model_jit.pt.ref    PyTorch reference output (logits + values)

The last two are for parity_check.cpp on the machine with LibTorch: the C++
engine must reproduce the same numbers within tolerance. Without that check the
engine could run very fast and play wrongly without anyone noticing.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dama import IN_PLANES, POLICY_SIZE          # noqa: E402
from model import PolicyValueNet                  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    weights_path, out_path = sys.argv[1], sys.argv[2]

    channels = int(os.environ.get("DAMA_CHANNELS", "96"))
    blocks = int(os.environ.get("DAMA_BLOCKS", "6"))
    use_se = os.environ.get("DAMA_USE_SE", "1") != "0"

    net = PolicyValueNet(channels=channels, n_blocks=blocks,
                         in_planes=IN_PLANES, use_se=use_se)
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        print(f"weights loaded from {weights_path}")
    else:
        print(f"WARNING: {weights_path} does not exist -> RANDOM network "
              "(fine only for the parity check, not for playing)")
    net.eval()

    # trace with batch > 1 so the batch dimension stays dynamic
    example = torch.zeros(4, IN_PLANES, 8, 8)
    with torch.no_grad():
        traced = torch.jit.trace(net, example)
        traced = torch.jit.freeze(traced)
    traced.save(out_path)
    print(f"TorchScript model saved to {out_path} "
          f"(network {channels}x{blocks}, in_planes={IN_PLANES})")

    # test input + reference for the parity check
    rng = np.random.default_rng(20260728)
    B = 8
    x = rng.random((B, IN_PLANES, 8, 8), dtype=np.float32)
    x.tofile(out_path + ".input")

    # STRICT fp32 for the parity reference: on Ampere+ GPUs convolutions use
    # TF32 (10 bits of mantissa) and the two sides can differ by ~1e-3 on the
    # logits from arithmetic alone. That noise masks the real question --
    # whether C++ and Python compute the SAME thing -- so the comparison runs in
    # full fp32 on both sides (parity_check.cpp calls set_tf32(false) before the
    # forward pass). In production TF32 stays on: it is much faster and does not
    # affect play.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    def reference_on(dev: str) -> np.ndarray:
        """PyTorch reference computed on `dev`, loading THE SAVED FILE.

        A delicate point: re-tracing the model on the device instead of
        reloading the one written to disk is wrong. Freezing FOLDS BatchNorm
        into the convolutions, and that computation run in fp32 on CPU or on
        CUDA gives slightly different weights; propagated through the residual
        blocks they gave ~1e-3 of relative difference on the logits. The
        comparison measured two networks with differently folded weights, not
        two implementations.

        Loading the same file the C++ loads, the only possible difference is the
        arithmetic of the execution -- which is exactly what the parity check
        must measure.
        """
        with torch.no_grad():
            t = torch.jit.load(out_path, map_location=dev)
            lo, va, _ = t(torch.from_numpy(x).to(dev))
        return np.concatenate([lo.float().cpu().numpy().reshape(-1),
                               va.float().cpu().numpy().reshape(-1)]).astype(np.float32)

    ref_device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    ref = reference_on(ref_device)
    ref.tofile(out_path + ".ref")
    net.to("cpu")
    print(f"parity reference saved: {out_path}.input / .ref "
          f"(B={B}, policy={POLICY_SIZE}, computed on {ref_device.upper()})")

    # How much do CPU and GPU differ from each other, in PyTorch ITSELF? It is
    # the yardstick that makes the parity_check result interpretable: the two
    # devices run different kernels, so a small difference between them is
    # NORMAL and indicates no error. Without this number "the C++ is wrong"
    # cannot be told apart from "the two devices compute differently".
    if ref_device == "cpu" and torch.cuda.is_available():
        cuda_ref = reference_on("cuda")
        net.to("cpu")
        d = float(np.max(np.abs(ref - cuda_ref)))
        print(f"\n  PyTorch CPU vs PyTorch CUDA, same model: max |diff| = {d:.3e}")
        print("  (this is the INTRINSIC variation between the two devices:")
        print("   the C++ backend on CUDA cannot do better than this against a")
        print("   CPU reference. For a strict comparison generate the")
        print(f"   reference on the same device: export_jit.py ... cuda)")

    print("\non the machine with LibTorch, now run:")
    print(f"  ./engine_c/build/parity_check {out_path} {ref_device}")


if __name__ == "__main__":
    main()
