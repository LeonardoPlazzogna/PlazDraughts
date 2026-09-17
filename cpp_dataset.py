"""
Reader for the dataset produced by the C++ engine (engine_c/dataset.hpp).

Training stays in PyTorch/Python: the C++ engine only plays the games and writes
the samples, and this module reads them back in the form
`train.train_on_samples` already expects -- a list of (X, pi, mask, z) tuples.
So the C++ path and the Python path feed the very same training, with no change
to train.py.

Format (little-endian):
    "DAMA" | version int32 | n_samples int32 | in_planes int32 | policy int32
    then n_samples records: x[in_planes*64] pi[policy] mask[policy] z[1], float32
"""
from __future__ import annotations
import numpy as np

MAGIC = b"DAMA"


def load_dataset(path):
    """Returns a list of (X[in_planes,8,8], pi[policy], mask[policy], z), the
    same form produced by selfplay.play_game in Python."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"{path}: unexpected magic {magic!r} (not a dama dataset)")
        header = np.frombuffer(f.read(16), dtype="<i4")
        version, n, in_planes, policy = (int(v) for v in header)
        if version != 1:
            raise ValueError(f"{path}: unsupported version {version}")
        rec = in_planes * 64 + 2 * policy + 1
        data = np.frombuffer(f.read(), dtype="<f4")

    if data.size != n * rec:
        raise ValueError(f"{path}: expected {n * rec} floats, found {data.size} "
                         "(truncated file?)")
    data = data.reshape(n, rec)

    off_pi = in_planes * 64
    off_mask = off_pi + policy
    off_z = off_mask + policy
    return [
        (data[i, :off_pi].reshape(in_planes, 8, 8).copy(),
         data[i, off_pi:off_mask].copy(),
         data[i, off_mask:off_z].copy(),
         float(data[i, off_z]))
        for i in range(n)
    ]


if __name__ == "__main__":
    import sys
    samples = load_dataset(sys.argv[1])
    print(f"{len(samples)} samples, X{samples[0][0].shape} "
          f"pi[{samples[0][1].shape[0]}] mask[{samples[0][2].shape[0]}]")
