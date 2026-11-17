"""
Dilated temporal convolutional network: per-step features in, per-step dead-ball logits out.

Convolutions are non-causal since matches are processed offline. There are three heads (ball dead,
stoppage start, stoppage end) plus an optional stoppage-type output.
"""

import torch
from torch import nn

HEADS = ("dead", "start", "stop")


class Block(nn.Module):
    """
    Two dilated convolutions with a residual connection, keeping the sequence length.
    """

    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation), nn.GELU(),
            nn.Dropout(dropout))
        self.norm = nn.GroupNorm(1, channels)                # layer norm over channels and time

    def forward(self, x):
        return self.norm(x + self.net(x))


class TCN(nn.Module):
    def __init__(self, n_inputs, channels, kernel, dilations, dropout, n_types=0, type_seed=1):
        super().__init__()
        self.inp = nn.Conv1d(n_inputs, channels, 1)
        self.blocks = nn.Sequential(*[Block(channels, kernel, d, dropout) for d in dilations])
        self.out = nn.Conv1d(channels, len(HEADS), 1)
        self.n_types = n_types
        if n_types:
            with torch.random.fork_rng(devices=[]):        # the global stream is left untouched
                torch.manual_seed(type_seed)
                self.type_out = nn.Conv1d(channels, n_types, 1)

    def forward(self, x):
        """
        Maps (batch, n_inputs, steps) to (batch, 3 + n_types, steps) logits: HEADS, then types.
        """
        h = self.blocks(self.inp(x))
        if self.n_types:
            return torch.cat([self.out(h), self.type_out(h)], dim=1)
        return self.out(h)


def receptive_field(kernel, dilations):
    """
    Number of steps a TCN output depends on.
    """
    return 1 + 2 * (kernel - 1) * sum(dilations)
