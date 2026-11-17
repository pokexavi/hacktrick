"""
Duration priors for the HSMM, learnt from labelled matches.

Stoppage lengths have several bumps (throw-ins, corners, goals), so the prior is a kernel
density on log-duration rather than a single log-normal. Lengths are in 5 Hz steps.
"""

import numpy as np


def run_lengths(state, period, scored):
    """
    Lengths of the complete True and False runs of a per-step bool series.
    Runs cut by the edge of a half or an unscored step are dropped, since their true length is unknown.
    """
    lengths = {True: [], False: []}
    change = np.flatnonzero((state[1:] != state[:-1]) | (period[1:] != period[:-1])
                            | (scored[1:] != scored[:-1])) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, len(state)]):
        complete = (scored[a] and a > 0 and b < len(state)
                    and period[a - 1] == period[a] == period[b] and scored[a - 1] and scored[b])
        if complete:
            lengths[bool(state[a])].append(int(b - a))
    return lengths


def log_duration_pmf(lengths, max_steps, bandwidth):
    """
    Log probability of each duration from 1 to max_steps, smoothed from the observed lengths.
    bandwidth is the width of each bump in log-duration.
    """
    d = np.arange(1, max_steps + 1)
    z = (np.log(d)[:, None] - np.log(np.asarray(lengths, dtype=float))[None, :]) / bandwidth
    # A Gaussian in log d is a log-normal in d: the 1/d turns the density in log d into one in d.
    density = np.exp(-0.5 * z ** 2).sum(axis=1) / d
    with np.errstate(divide="ignore"):
        return np.log(density / density.sum())
