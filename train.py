"""
Training loop for Italian draughts.

  loss = masked policy CE + WDL CE

  * policy: log-softmax over the LEGAL slots only (masking); target = the MCTS
    visit distribution. The gradient wastes no signal on impossible actions.
  * value: cross-entropy on the WDL head. The target is derived from the
    outcome z: win (z>0), draw (z==0), loss (z<0), softened when z is
    fractional (see below). Better calibrated on draws than an MSE on a scalar.
    The scalar `value` used by the MCTS stays DERIVED from the WDL in the model.

TRAINING DYNAMICS. Two choices that are not the obvious ones, and why:

  * Learning-rate SCHEDULE -- ON by default. Linear warmup + cosine decay
    within each call. The warmup avoids destabilizing the champion's weights in
    the first steps; the cosine lets the network settle towards the end instead
    of oscillating at a constant step size until the last batch. What tells the
    two apart is the mean arena score of the candidates, not the training loss.

  * Weight EMA -- OFF in the conductor (DAMA_EMA=0). The idea was to reduce the
    variance of the final weights, since the promotion gate is noisy, but it
    works against this regime: here every cycle is a SHORT fine-tuning that
    moves away from the champion's weights in a definite direction. Averaging a
    directional trajectory means lagging behind, so the candidate comes out
    weaker than the champion and the arena rejects it. EMA helps when the
    trajectory oscillates around a minimum, not when it advances -- which is
    why the chess project uses it (long training in a single phase) and this
    one does not. It stays available with DAMA_EMA=1, to re-evaluate it if the
    regime changes. Note that the function's own default is use_ema=True:
    callers that do not pass it (the benchmark, some tools and tests) train
    with the EMA on.

    If it is turned back on: the decay must NOT be set to 0.999. With a few
    hundred steps per cycle, a 1000-step window would anchor the EMA to the
    initial weights and the candidate would come out identical to the
    champion. The default uses a window of a quarter of the actual steps.
"""
from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

NEG = -1e9  # for masking: excludes the illegal actions from the softmax


def _lr_factor(step: int, total: int, warmup_frac: float, final_frac: float) -> float:
    """Learning-rate multiplier at step `step` out of `total`.

    Linear warmup from 0, then cosine from 1 to `final_frac` (not to zero: the
    last batches must still contribute something)."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    if total <= warm:
        return 1.0
    prog = (step - warm) / max(1, total - warm)
    return final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))


def train_on_samples(net, samples, epochs: int = 1, batch_size: int = 256,
                     lr: float = 1e-3, weight_decay: float = 1e-4,
                     device: str = "cpu", warmup_frac: float = 0.05,
                     lr_final_frac: float = 0.1, use_ema: bool = True,
                     ema_window_frac: float = 0.25, optimizer: str = "adam"):
    """Trains `net` in place on the (X, pi, mask, z) samples. Returns metrics."""
    if len(samples) == 0:
        return {"n": 0}

    X = torch.from_numpy(np.stack([s[0] for s in samples]))
    P = torch.from_numpy(np.stack([s[1] for s in samples]))
    M = torch.from_numpy(np.stack([s[2] for s in samples]))
    Z = torch.from_numpy(np.array([s[3] for s in samples], np.float32)).unsqueeze(1)

    loader = DataLoader(TensorDataset(X, P, M, Z), batch_size=batch_size, shuffle=True)
    # Adam applies weight decay THROUGH the gradient, so its effective strength
    # is rescaled by the adaptive denominator and is no longer uniform across
    # parameters; AdamW applies it to the weights directly. With a non-zero
    # weight_decay the two do NOT coincide. The default stays "adam" so as not
    # to silently change the behavior of the runs already made: which of the
    # two plays better is settled with tools/sweep_train.py.
    if optimizer == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer must be 'adam' or 'adamw', not " + repr(optimizer))

    net.to(device).train()

    total_steps = max(1, epochs * len(loader))
    # EMA window proportional to the actual number of steps (see the module notes)
    ema_decay = None
    ema = None
    if use_ema and total_steps >= 8:
        window = max(2.0, total_steps * ema_window_frac)
        ema_decay = 1.0 - 1.0 / window
        ema = {k: v.detach().clone().float()
               for k, v in net.state_dict().items() if v.dtype.is_floating_point}

    tot = {"loss": 0.0, "policy": 0.0, "value": 0.0, "batches": 0}
    step = 0
    _checked = False        # mask/target invariant, see below
    for _ in range(epochs):
        for xb, pb, mb, zb in loader:
            for g in opt.param_groups:
                g["lr"] = lr * _lr_factor(step, total_steps, warmup_frac, lr_final_frac)

            xb, pb, mb, zb = xb.to(device), pb.to(device), mb.to(device), zb.to(device)
            policy_logits, _value, wdl_logits = net(xb)

            # policy CE masked to the legal actions
            masked = policy_logits.masked_fill(mb == 0, NEG)
            logp = F.log_softmax(masked, dim=1)

            # INVARIANT: the target must have no mass where the mask says
            # "illegal". If it does, that mass multiplies a logit pushed to
            # -1e9 and the loss jumps to ~1e8: training blows up, and the
            # symptom -- a huge loss -- sends one looking for the fault in the
            # learning rate or in bad data, not in the mismatch between mask
            # and target that caused it.
            #
            # Checked ONCE per call, on the first batch: it costs one sum over
            # a batch and covers the case that matters, a sample producer that
            # changes convention. Verified on both producers (Python self-play
            # and the C++ dataset).
            if not _checked:
                off_mask = float((pb * (mb == 0)).sum())
                if off_mask > 1e-6:
                    raise ValueError(
                        f"policy target inconsistent with the mask: "
                        f"{off_mask:.4f} of probability on moves declared "
                        f"illegal. This is not a training problem, it is a "
                        f"sample problem: whoever produces them uses a "
                        f"different convention from whoever builds the mask.")
                _checked = True

            policy_loss = -(pb * logp).sum(dim=1).mean()

            # WDL cross-entropy with a SOFT target.
            #
            # Taking the class from the SIGN of z alone, cls = (z<=0) + (z<0),
            # is correct as long as z is exactly -1, 0 or +1, but it hides any
            # intermediate label -- and so it made the value discount a no-op:
            # two networks trained with and without the discount came out
            # bit-for-bit identical.
            #
            # Here z becomes the unique distribution over (win, draw, loss)
            # with mean z that puts the rest on the draw:
            #     p_win = max(z,0)   p_loss = max(-z,0)   p_draw = 1-|z|
            # With an integer z the target is one-hot again and the behavior is
            # identical to the sign rule: the change is inert until the
            # discount is turned on.
            z = zb.squeeze(1)
            p_win = z.clamp(min=0.0)
            p_loss = (-z).clamp(min=0.0)
            target = torch.stack([p_win, 1.0 - p_win - p_loss, p_loss], dim=1)
            value_loss = -(target * F.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()

            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            if ema is not None:
                with torch.no_grad():
                    sd = net.state_dict()
                    for k in ema:
                        ema[k].mul_(ema_decay).add_(sd[k].detach().float(),
                                                    alpha=1.0 - ema_decay)

            tot["loss"] += loss.item()
            tot["policy"] += policy_loss.item()
            tot["value"] += value_loss.item()
            tot["batches"] += 1
            step += 1

    # The averaged weights become the candidate's: this is the version that
    # goes to the arena and, if promoted, becomes the new champion.
    if ema is not None:
        with torch.no_grad():
            sd = net.state_dict()
            for k, v in ema.items():
                sd[k].copy_(v.to(sd[k].dtype))

    b = max(1, tot["batches"])
    return {"n": len(samples), "loss": tot["loss"] / b,
            "policy": tot["policy"] / b, "value": tot["value"] / b,
            "steps": total_steps, "lr_end": lr * _lr_factor(total_steps - 1, total_steps,
                                                            warmup_frac, lr_final_frac),
            "ema_decay": ema_decay if ema_decay is not None else 0.0}
