"""
Policy-value network for Italian draughts.

  * COMPACT policy head: a flat [B, 256] output (32 dark squares x 8 slots =
    4 directions x {step, jump}) instead of a from-to head of 64x64 = 4096
    outputs, about 99% of which are illegal in draughts. A 1x1 conv down to 8
    channels, reordered into [B, POLICY_SIZE] with index = 8*dark_square +
    2*direction + mode (the same as dama/encoder.py).
  * WDL value head: 3 logits (win/draw/loss) trained with cross-entropy; the
    scalar `value` in [-1, 1] is DERIVED from them (P(win) - P(loss) -
    contempt * P(draw)), which is better calibrated on draws.
  * No global average pooling in the value head: 1x1 conv -> flatten -> FC, so
    the 8x8 spatial structure (advancement, last row) is not collapsed.
  * Optional squeeze-and-excitation blocks (`use_se`), to cut latency when
    needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SLOTS = 8                       # 4 directions x 2 modes (step/jump)
# The policy covers only the 32 dark (playable) squares -> 32*8 = 256 (not 64*8 = 512).
_DARK_SQUARES = [s for s in range(64) if (s // 8 + s % 8) % 2 == 1]
POLICY_SIZE = len(_DARK_SQUARES) * SLOTS   # 256


class SqueezeExcitation(nn.Module):
    """Per-channel recalibration (SE): global pooling -> 2 FC -> sigmoid gate."""
    def __init__(self, ch: int, reduced: int | None = None):
        super().__init__()
        if reduced is None:
            reduced = max(ch // 8, 8)
        self.fc1 = nn.Linear(ch, reduced)
        self.fc2 = nn.Linear(reduced, ch)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    """Conv3x3-BN-ReLU-Conv3x3-BN-[SE]-(+x)-ReLU. SE is optional: it can be
    dropped when batch-1 latency matters."""
    def __init__(self, ch: int, use_se: bool = True):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.se = SqueezeExcitation(ch) if use_se else nn.Identity()

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        y = self.se(y)
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """Input [B, in_planes, 8, 8]. Output: (policy_logits[B,POLICY_SIZE],
    value[B,1] in [-1,1], wdl_logits[B,3])."""
    def __init__(self, channels: int = 64, n_blocks: int = 6,
                 in_planes: int = 7, contempt: float = 0.0,
                 use_se: bool = True, v_channels: int = 16,
                 v_hidden: int | None = None):
        super().__init__()
        self.in_planes = in_planes
        self.contempt = float(contempt)

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels, use_se)
                                     for _ in range(n_blocks)])

        # --- compact policy head: 8 channels = 8 slots per square ---
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, SLOTS, 1, bias=True),   # [B, 8, 8, 8]
        )
        # (r, c) of the 32 dark squares, to keep only those in the output
        self.register_buffer("dark_r", torch.tensor([s // 8 for s in _DARK_SQUARES]))
        self.register_buffer("dark_c", torch.tensor([s % 8 for s in _DARK_SQUARES]))

        # --- value head -> WDL, WITHOUT global average pooling: 1x1 conv ->
        #     flatten -> FC, so the 8x8 spatial structure is preserved instead
        #     of being collapsed into a vector. ---
        # The head has TWO possible bottlenecks, both configurable:
        #   v_channels : the 1x1 conv compresses `channels` channels into
        #                v_channels (e.g. 96 -> 16) before the flatten
        #   v_hidden   : the width of the hidden layer before the 3 WDL logits
        #                (defaults to `channels`)
        # They matter because, with the policy at the entropy floor of its
        # target, almost all the progress goes through the value head: it is
        # what separates the moves in PUCT and decides the games.
        v_hidden = channels if v_hidden is None else v_hidden
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, v_channels, 1, bias=False),
            nn.BatchNorm2d(v_channels), nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(v_channels * 64, v_hidden)
        self.wdl_fc = nn.Linear(v_hidden, 3)           # logits [win, draw, loss]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)

        # policy: [B,8,8,8] -> keep the 32 dark squares -> [B,32,8] -> [B,256]
        # (index = dark_index*8 + slot, as in the encoder)
        p = self.policy_head(x)
        p = p[:, :, self.dark_r, self.dark_c]           # [B, 8, 32]
        policy_logits = p.permute(0, 2, 1).reshape(x.size(0), POLICY_SIZE)

        h = self.v_conv(x).flatten(1)
        h = F.relu(self.v_fc1(h))
        wdl_logits = self.wdl_fc(h)                    # [B, 3]
        wdl_p = F.softmax(wdl_logits, dim=1)
        value = wdl_p[:, 0:1] - wdl_p[:, 2:3] - self.contempt * wdl_p[:, 1:2]
        value = torch.clamp(value, -1.0, 1.0)          # [B, 1]

        return policy_logits, value, wdl_logits
