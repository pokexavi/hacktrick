"""
Make DFL's dead ball move, as it does in BAR-ATH.

DFL freezes the ball during dead time while BAR-ATH's keeps moving, so a model could learn that a
still ball means dead play. Here the ball goes from where it froze to the restart spot with a
random wander, scaled to a target median speed.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d

from external import schema

# Params
TARGET_SPEED = 2.7       # m/s: median ball speed on BAR-ATH's tracked public GT dead frames
SMOOTH_S = 1.0           # the wander's velocity changes over about this long
MARGIN_M = 3.0           # the wandering ball stays within this of the pitch lines


def _wander(n, rng, smooth_s=SMOOTH_S):
    """
    Random (n, 2) displacement with unit median speed, zero at both ends.
    """
    v = gaussian_filter1d(rng.normal(size=(n, 2)), smooth_s * schema.FPS, axis=0)
    v /= max(np.median(np.linalg.norm(v, axis=1)) * schema.FPS, 1e-9)
    d = np.cumsum(v, axis=0)
    return d - np.linspace(0, 1, n)[:, None] * d[-1]


def unfreeze(match, label, rng, target=TARGET_SPEED, smooth_s=SMOOTH_S, margin_m=MARGIN_M):
    """
    Copy of the match with the ball moving through every dead stretch of label.
    """
    ball = match.ball.copy()
    limit = np.asarray(match.pitch, dtype=np.float32) / 2 + margin_m
    dead = label == "D"
    edges = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b >= len(ball) or b - a < 3 or not np.isfinite(ball[[a, b], 0]).all():
            continue
        n = b - a
        line = ball[a, :2] + np.linspace(0, 1, n, endpoint=False)[:, None] * (ball[b, :2] - ball[a, :2])
        wander = _wander(n, rng, smooth_s)
        ball[a:b, :2] = np.clip(line + _scale(line, wander, target) * wander, -limit, limit)
    return replace(match, ball=ball, meta=dict(match.meta, ball_unfrozen=True))


def _median_speed(path):
    return np.median(np.linalg.norm(np.diff(path, axis=0), axis=1)) * schema.FPS


def _scale(line, wander, target, steps=20):
    """
    Wander scale that brings the path's median speed to target, 0 if the straight line is already faster.
    """
    if _median_speed(line) >= target:
        return 0.0
    low, high = 0.0, 4.0 * target
    for _ in range(steps):
        mid = (low + high) / 2
        if _median_speed(line + mid * wander) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def from_config(section):
    """
    Unfreeze function from a config's [augment.dead_ball.uniform] section.
    """
    return lambda match, label, rng: unfreeze(match, label, rng, section["target_speed"],
                                              section["smooth_s"], section["margin_m"])
