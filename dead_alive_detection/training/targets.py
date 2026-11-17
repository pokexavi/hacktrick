"""
Training targets of the TCN's three heads, and random training windows.
Kept free of torch so the decoder can use the same targets.
"""

import numpy as np


def targets(t, sigma_steps):
    """
    Targets for the dead, start and stop heads, and per-step weights (1 = scored).
    Start and stop are soft Gaussian bumps around the true boundaries.
    """
    dead, period = t["dead"], t["period"]
    same_half = np.r_[False, period[1:] == period[:-1]]
    starts = (dead & ~np.r_[False, dead[:-1]] & same_half).astype(float)
    stops = (~dead & np.r_[False, dead[:-1]] & same_half).astype(float)
    reach = int(3 * sigma_steps)
    bump = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma_steps) ** 2)
    soft = [np.minimum(1.0, np.convolve(edges, bump, mode="same")) for edges in (starts, stops)]
    return (np.stack([dead.astype(float), *soft]).astype(np.float32),
            t["scored"].astype(np.float32))


def windows(period, length, rng):
    """
    Random windows of `length` steps inside each half, about one per window length of the half.
    """
    out = []
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if len(rows) > length:
            starts = rng.integers(rows[0], rows[-1] + 1 - length, size=len(rows) // length)
            out += [(int(a), int(a) + length) for a in starts]
    return out
