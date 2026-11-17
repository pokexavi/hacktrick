"""
Simulated referee whistles on a DFL match, as BAR-ATH's audio detector would report them.

DFL has no audio, so whistles are placed at the start and restart of true stoppages depending on
their type, kept with some recall, plus random false whistles. Whistle lengths come from the
detector's output on BAR-ATH.
"""

import numpy as np
import pandas as pd

FPS = 25
STEP = 5        # frames per 5 Hz step (loading/tensors.STEP)


def read_durations(path):
    """
    Lengths in seconds of the whistles the detector found.
    """
    table = pd.read_csv(path)
    if not len(table):
        raise ValueError("no whistles in {}: run audio/whistle.py first".format(path))
    return table["duration_s"].to_numpy(float)


def simulate(stoppages, period, section, profile, durations, rng):
    """
    (start, end) whistle frames on one match, end exclusive.
    """
    recall = rng.uniform(*profile["recall"])
    rate = rng.uniform(*profile["false_per_min"])
    mean, sd, clip = section["start_offset_s"]
    lead_lo, lead_hi = section["restart_lead_s"]
    out = []
    for a, b, kind in stoppages:
        # 8 draws per stoppage whatever happens: referee and detector, start and restart
        u = rng.random(4)
        offset = float(np.clip(mean + sd * rng.standard_normal(), -clip, clip))
        lead = rng.uniform(lead_lo, lead_hi)
        length = durations[rng.integers(len(durations), size=2)]
        if u[0] < section["start"].get(kind, 0.0) and u[1] < recall:
            s = a + offset * FPS
            out.append((s, s + length[0] * FPS))
        if u[2] < section["restart"].get(kind, 0.0) and u[3] < recall:
            s = max(b - lead * FPS, a)
            out.append((s, s + length[1] * FPS))
    in_half = np.flatnonzero(np.asarray(period) > 0)
    minutes = len(in_half) / FPS / 60
    n_false = rng.poisson(rate * minutes)
    for f, length in zip(in_half[rng.integers(len(in_half), size=n_false)],
                         durations[rng.integers(len(durations), size=n_false)]):
        out.append((float(f), float(f) + length * FPS))
    return sorted(out)


def to_steps(whistles, rows, step=STEP):
    """
    Per-step flag: 1 where a whistle sounds during the step, else 0.
    """
    rows = np.asarray(rows)
    flag = np.zeros(len(rows), np.float32)
    for s, e in whistles:
        lo = np.searchsorted(rows, s - step, side="right")     # first step with row + step > s
        hi = np.searchsorted(rows, max(e, s + 1e-6), side="left")   # steps with row < e
        flag[lo:hi] = 1.0
    return flag


def detected(table, frames):
    """
    (start, end) row indices of the detector's whistles in a match.
    """
    frames = np.asarray(frames)
    start = table["frame"].to_numpy(float)
    end = start + table["duration_s"].to_numpy(float) * FPS
    lo = np.searchsorted(frames, start)
    return [(float(a), float(a + (e - s))) for a, s, e in zip(lo, start, end)
            if a < len(frames)]
