"""
Calibration of the TRAINING parameters on data already collected.

WHY IT EXISTS. The obvious protocol for comparing two learning rates is to run
two short runs and see which champion is stronger. It is wrong, or rather: it is
a confounded experiment. Each of the two runs generates its OWN self-play, so
the two final networks differ for two reasons at once -- the parameter and the
games seen -- and the comparison does not separate the two. With ten cycles the
variance between two sets of games is easily larger than the effect one wants
to measure.

Here the buffer of the run is frozen once and all the candidates are trained ON
THE SAME samples, starting from the SAME initial champion. The only difference
between two candidates is the parameter. It is a paired design: more precise,
and incidentally much faster, because it generates no games.

WHAT IT MEASURES AND WHAT IT DOES NOT. The metrics computed here are on held-out
data, never seen in training, and say how closely each candidate follows its own
targets. They do NOT say which one plays better: a network can fit the visit
distribution better and play worse. That takes compare.py, and the script prints
the ready-made commands at the end.

THE EMA IS ON. The candidates are trained with the defaults of train_on_samples,
which keep the weight EMA on (use_ema=True), while the conductor turns it off
(DAMA_EMA=0, see train.py). The figures therefore describe training WITH the
EMA; `python tools/sweep_train.py use_ema 1 0` measures how much that matters.

USAGE
    python tools/sweep_train.py lr 1e-3 3e-4 1e-4
    python tools/sweep_train.py optimizer adam adamw
    python tools/sweep_train.py epochs 1 2 4
    python tools/sweep_train.py weight_decay 1e-4 0

    DAMA_WEIGHTS_DIR=weights   where champion.pt and train_state.pkl are
    DAMA_SWEEP_OUT=sweep       where to write the candidates
    DAMA_SWEEP_HOLDOUT=0.1     fraction of the samples held out
"""
from __future__ import annotations
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# The project modules live in the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dama import IN_PLANES
from model import PolicyValueNet
from train import train_on_samples

WEIGHTS  = os.environ.get("DAMA_WEIGHTS_DIR", "weights")
OUT      = os.environ.get("DAMA_SWEEP_OUT", "sweep")
HOLDOUT  = float(os.environ.get("DAMA_SWEEP_HOLDOUT", "0.1"))
CHANNELS = int(os.environ.get("DAMA_CHANNELS", "96"))
N_BLOCKS = int(os.environ.get("DAMA_BLOCKS", "6"))
USE_SE   = os.environ.get("DAMA_USE_SE", "1") != "0"
EPOCHS   = int(os.environ.get("DAMA_EPOCHS", "2"))
BATCH    = int(os.environ.get("DAMA_BATCH", "1024"))
LR       = float(os.environ.get("DAMA_LR", "1e-3"))
DEVICE   = os.environ.get("DAMA_DEVICE", "cpu")
SEED     = int(os.environ.get("DAMA_SWEEP_SEED", "20260801"))

NEG = -1e9

# accepted parameters, with their converter
KNOBS = {
    "lr": float, "weight_decay": float, "epochs": int, "batch_size": int,
    "optimizer": str, "use_ema": lambda v: v not in ("0", "false", "no"),
    "warmup_frac": float, "lr_final_frac": float,
}


@torch.no_grad()
def evaluate(net, samples, device: str) -> dict:
    """Metrics on held-out samples, with the entropy decomposition.

    The policy loss cannot go below the entropy of the target: the difference
    between the two IS the divergence between network and search. Reporting
    them separately avoids confusing "the network does not follow the target"
    with "the target is spread out", which call for opposite remedies.
    """
    net.to(device).eval()
    X = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
    P = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
    M = torch.from_numpy(np.stack([s[2] for s in samples])).to(device)
    Z = torch.from_numpy(np.array([s[3] for s in samples], np.float32)).to(device)

    pol = val = ent = 0.0
    n = 0
    for i in range(0, len(X), 4096):
        xb, pb, mb, zb = X[i:i+4096], P[i:i+4096], M[i:i+4096], Z[i:i+4096]
        logits, _v, wdl = net(xb)
        logp = torch.log_softmax(logits.masked_fill(mb == 0, NEG), dim=1)
        pol += float(-(pb * logp).sum(dim=1).sum())
        ent += float(-(pb * torch.log(pb.clamp_min(1e-12))).sum(dim=1).sum())
        # The logits of the head are in the order [win, draw, loss] (model.py),
        # so z=+1 must land on class 0. The previous version wrote z.sign()+1,
        # which sends +1 to class 2 -- that is, LOSS -- and -1 to class 0. The
        # classes were swapped, and the consequence was misleading in the worst
        # way: the more the model learned, the worse the loss looked. On a real
        # buffer it gave 4.0 at two epochs and 6.4 at thirty, when pure chance
        # over three classes is worth 1.10 -- a model cannot be four times worse
        # than chance on data it has just seen, and that number had to be read
        # as a defect of the measurement.
        cls = ((zb <= 0).long() + (zb < 0).long())   # +1,0,-1 -> 0,1,2
        val += float(F.cross_entropy(wdl, cls, reduction="sum"))
        n += len(xb)

    pol, val, ent = pol / n, val / n, ent / n
    return {"policy": pol, "value": val, "entropy": ent, "kl": pol - ent}


def main():
    # A --help that answers: these tools take no arguments, so without this
    # line they would start computing (see env_help.py).
    from env_help import help_if_requested
    help_if_requested(__doc__, __file__)
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    knob, values = sys.argv[1], sys.argv[2:]
    if knob not in KNOBS:
        raise SystemExit(f"parameter '{knob}' not accepted; choose among: "
                         + ", ".join(sorted(KNOBS)))
    conv = KNOBS[knob]
    values = [conv(v) for v in values]

    ckpt = os.path.join(WEIGHTS, "champion.pt")
    state = os.path.join(WEIGHTS, "train_state.pkl")
    for p in (ckpt, state):
        if not os.path.exists(p):
            raise SystemExit(f"{p} is missing: a run that has completed at least "
                             "one cycle is needed")

    print(f"[sweep] loading the buffer from {state} ...")
    with open(state, "rb") as f:
        buffer = list(pickle.load(f)["buffer"])
    if not buffer:
        raise SystemExit("the buffer is empty")

    # deterministic split: the same held-out samples for every candidate,
    # otherwise the comparisons would not be paired
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(buffer))
    n_hold = max(1, int(len(buffer) * HOLDOUT))
    hold = [buffer[i] for i in idx[:n_hold]]
    train = [buffer[i] for i in idx[n_hold:]]
    print(f"[sweep] {len(train)} training samples, {len(hold)} held out")
    print(f"[sweep] parameter: {knob} in {values}")
    print(f"[sweep] every candidate starts from {ckpt} and sees the same data\n")

    os.makedirs(OUT, exist_ok=True)
    base = torch.load(ckpt, map_location="cpu", weights_only=True)

    print(f"  {knob:<14} {'loss_pol':>8} {'entropy':>8} {'KL':>6} {'KL_train':>8} "
          f"{'gap':>7} {'loss_val':>8}  {'time':>5}")
    print("  " + "-" * 78)
    produced = []
    for v in values:
        net = PolicyValueNet(in_planes=IN_PLANES, channels=CHANNELS,
                                     n_blocks=N_BLOCKS, use_se=USE_SE)
        net.load_state_dict(base)
        torch.manual_seed(SEED)          # same shuffle initialization

        kw = dict(epochs=EPOCHS, batch_size=BATCH, lr=LR, device=DEVICE)
        kw[knob] = v
        t0 = time.time()
        train_on_samples(net, train, **kw)
        dt = time.time() - t0

        m = evaluate(net, hold, DEVICE)
        # The same metrics on the data SEEN in training. On their own the
        # held-out ones do not tell apart the two opposite ways of failing: a
        # network that cannot even memorize what it sees (not enough capacity)
        # and one that memorizes very well and does not generalize (not enough
        # data) show the same held-out KL. The gap between the two columns is
        # what separates them, and it points to opposite levers -- more channels
        # against more games.
        m_tr = evaluate(net, train[:len(hold)], DEVICE)
        name = os.path.join(OUT, f"{knob}_{v}.pt".replace("/", "_"))
        torch.save(net.state_dict(), name)
        produced.append(name)
        print(f"  {str(v):<14} {m['policy']:8.4f} {m['entropy']:8.4f} "
              f"{m['kl']:6.4f} {m_tr['kl']:8.4f} {m['kl']-m_tr['kl']:7.4f} "
              f"{m['value']:8.4f} {dt:5.0f}s")

    print("\n  The metrics above are on HELD-OUT data and measure how closely")
    print("  each candidate follows its own targets. They do NOT measure playing")
    print("  strength: a network can follow them better and play worse.")
    print("\n  To rank them by strength, play them against each other:")
    for a, b in zip(produced, produced[1:]):
        print(f"    python tools/compare.py {a} {b}")


if __name__ == "__main__":
    main()
