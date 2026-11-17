"""
Turns one processed match (DFL or BAR-ATH) into the 5 Hz per-step arrays the model reads.

The home team always attacks +x, positions are scaled so the lines sit at +/-1, and velocities
stay in m/s. Missing values are 0, with a mask saying what is real. Players are slots, not people.
"""

import numpy as np

from dead_alive_detection.loading.clean import mask_edge_artifacts
from external import schema

# Params
STEP = 5        # frames per step: 25 fps -> 5 Hz
STEPS_PER_S = schema.FPS / STEP
SPAN = 5        # velocity = displacement between t - SPAN and t + SPAN frames (0.4 s), see velocity()


def velocity(xy, period, span=SPAN):
    """
    Velocity in m/s by central difference over +/- span frames, which smooths tracking jitter.
    NaN across gaps and across halftime.
    """
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[span:-span] = (xy[2 * span:] - xy[:-2 * span]) * (schema.FPS / (2.0 * span))
    same_period = np.zeros(len(period), dtype=bool)
    same_period[span:-span] = period[2 * span:] == period[:-2 * span]
    v[~same_period] = np.nan
    return v


def step_rows(period, step=STEP):
    """
    25 fps frame indices of the 5 Hz steps, counted from the start of each half.
    """
    return np.concatenate([np.flatnonzero(period == p)[::step] for p in (1, 2)])


def tokens(xy, period, rows, scale):
    """
    Scaled position, velocity and presence mask of the ball or a team, at the steps.
    """
    present = np.isfinite(xy[rows, ..., 0])
    features = np.concatenate([xy[rows] / scale, velocity(xy, period)[rows]], axis=-1)
    return np.nan_to_num(features).astype(np.float32), present


def match_tensors(match):
    """
    The whole match as a dict of per-step arrays: positions, masks, labels and pitch size.
    """
    rows = step_rows(match.period)
    scale = np.array(match.pitch, dtype=np.float32) / 2
    label = mask_edge_artifacts(match.label, match.period)[rows]
    ball, ball_seen = tokens(match.ball[:, :2], match.period, rows, scale)
    home, home_mask = tokens(match.home, match.period, rows, scale)
    away, away_mask = tokens(match.away, match.period, rows, scale)
    return {"row": rows, "period": match.period[rows],
            "ball": ball, "ball_seen": ball_seen,
            "home": home, "home_mask": home_mask,
            "away": away, "away_mask": away_mask,
            "dead": label == "D", "scored": label != "X", "half_pitch": scale}


def to_frames(values, rows, n_frames):
    """
    Maps per-step values back to 25 fps: each frame takes the last step at or before it.
    """
    latest = np.searchsorted(rows, np.arange(n_frames), side="right") - 1
    return values[np.maximum(latest, 0)]
