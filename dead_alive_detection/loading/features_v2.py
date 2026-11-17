"""
Feature set v2 of the TCN, chosen with feature_set = "v2" in the config's [features].

Starts from v1 (loading/features.py), drops a few redundant columns, reads the speed features only
on valid player motion so tracker slot swaps don't look like sprints, and adds keeper features.
"""

import numpy as np

from dead_alive_detection.loading import features as v1
from dead_alive_detection.loading.tensors import STEPS_PER_S

# Physics, not hyperparameters
MAX_SPEED_MPS = 12.0      # faster than any player: the slot changed person
MAX_ACCEL_MPS2 = 8.0      # likewise for a velocity change over 0.4 s
MAX_BALL_MPS = 35.0       # a faster ball step is a tracker jump, not a kick
TRIM_TOP = 0.10           # robust mean_speed leaves out the fastest 10% of the valid players
SPEED_CHANGE_S = 1.0      # speed_change: central difference over +/- this
KEEPER_LINE_M = 20.0      # a keeper is at most this far from its own goal line...
KEEPER_HALF_WIDTH_M = 25.0  # ...and this far from the long axis of the pitch

PLAYER_NAMES = ("mean_speed", "walking_frac", "speed_change", "home_spread", "away_spread",
                "centroid_gap", "box_frac", "touchline_frac")
BALL_NAMES = ("ball_seen", "since_seen_s", "until_seen_s", "goal_line_dist", "touchline_dist",
              "centre_dist", "ball_speed", "ball_kick", "still_s", "nearest_player")
COLLECTIVE_NAMES = ("p90_speed", "running_frac", "stop_drop", "restart_rise", "stopped_s",
                    "stopped_ahead_s", "decel_frac")
NEW_NAMES = ("valid_motion_frac", "median_valid_speed", "ball_gk_dist", "last_ball_in_box")
NAMES = PLAYER_NAMES + BALL_NAMES + COLLECTIVE_NAMES + v1.LAST_SEEN_NAMES + NEW_NAMES

# Columns of v1's blocks that v2 keeps (by name, so a reordering of v1 cannot slip through).
_BALL_KEEP = [v1.BALL_NAMES.index(n) for n in BALL_NAMES]
_COLLECTIVE_KEEP = [v1.COLLECTIVE_NAMES.index(n) for n in COLLECTIVE_NAMES]


def _players(t):
    """
    Player positions in metres, velocities in m/s and presence, per slot.
    """
    half = t["half_pitch"]
    xy = np.concatenate([t["home"][..., :2], t["away"][..., :2]], axis=1) * half
    v = np.concatenate([t["home"][..., 2:], t["away"][..., 2:]], axis=1).astype(float)
    mask = np.concatenate([t["home_mask"], t["away_mask"]], axis=1)
    return xy.astype(float), v, mask


def valid_motion(t):
    """
    Speed, presence and valid-motion mask of every slot.
    Motion is valid when speed, step displacements and acceleration are all physically plausible.
    """
    xy, v, mask = _players(t)
    period = t["period"]
    n = len(period)
    speed = np.linalg.norm(v, axis=-1)
    linked = np.zeros(mask.shape, bool)                  # present at t and at t + 1, same half
    linked[:-1] = mask[:-1] & mask[1:] & (period[:-1] == period[1:])[:, None]
    jump = np.full(mask.shape, np.inf)                   # speed of the displacement t -> t + 1
    jump[:-1] = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * STEPS_PER_S
    jump = np.where(linked, jump, np.inf)
    back = np.full(mask.shape, np.inf)                   # t - 1 -> t
    back[1:] = jump[:-1]
    accel = np.full(mask.shape, np.inf)                  # velocity change t - 1 -> t + 1
    if n > 2:
        accel[1:-1] = np.linalg.norm(v[2:] - v[:-2], axis=-1) * STEPS_PER_S / 2
    with np.errstate(invalid="ignore"):
        valid = (mask & (speed <= MAX_SPEED_MPS) & (back <= MAX_SPEED_MPS)
                 & (jump <= MAX_SPEED_MPS) & (accel <= MAX_ACCEL_MPS2))
    return speed, mask, valid


def _present_for(mask, period):
    """
    Per slot, how many steps in a row it has been present, within each half.
    """
    out = np.zeros(mask.shape)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if not len(rows):
            continue
        m = mask[rows]
        index = np.arange(len(rows))[:, None]
        last_absent = np.maximum.accumulate(np.where(~m, index, -1), axis=0)
        out[rows] = np.where(m, index - last_absent, 0)
    return out


def _central_rate(x, period, half_width):
    """
    Per-second change of x across +/- half_width steps, within each half.
    """
    out = np.full(len(x), np.nan)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        k = half_width
        if len(rows) > 2 * k:
            values = x[rows]
            out[rows[k:-k]] = (values[2 * k:] - values[:-2 * k]) / (2 * k / STEPS_PER_S)
    return out


def robust_signals(t):
    """
    Per-step speed signals computed on valid motion only.
    """
    speed, mask, valid = valid_motion(t)
    period = t["period"]
    s = np.where(valid, speed, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), v1._quiet():
        p90 = np.nanpercentile(s, 90, axis=1)
        median = np.nanmedian(s, axis=1)
        top = np.nanpercentile(s, 100 * (1 - TRIM_TOP), axis=1)
        kept = valid & (speed <= top[:, None])
        mean = v1._masked_mean(speed, kept)
        valid_frac = np.where(mask.any(axis=1), valid.sum(axis=1) / mask.sum(axis=1), np.nan)
    # Braking: the same slot valid now and 1 s ago, present on every step in between.
    second = round(STEPS_PER_S)
    before = np.full(speed.shape, np.nan)
    before[second:] = speed[:-second]
    was_valid = np.zeros(valid.shape, bool)
    was_valid[second:] = valid[:-second]
    eligible = valid & was_valid & (_present_for(mask, period) > second)
    braking = v1._masked_mean(np.nan_to_num(before - speed) > v1.DECEL_MPS, eligible)
    return {"valid_frac": valid_frac, "median": median, "mean_speed": mean, "p90": p90,
            "running_frac": v1._masked_mean(speed > v1.RUN_MPS, valid), "spread": np.zeros(len(s)),
            "braking": braking}


def robust_ball_kick(t):
    """
    Highest plausible ball speed within +/- KICK_S, ignoring tracker jumps.
    """
    half, period, seen = t["half_pitch"], t["period"], t["ball_seen"]
    xy = t["ball"][:, :2].astype(float) * half
    speed = np.linalg.norm(t["ball"][:, 2:].astype(float), axis=-1)
    step = np.full(len(seen), np.inf)                    # t -> t + 1
    linked = seen[:-1] & seen[1:] & (period[:-1] == period[1:])
    step[:-1] = np.where(linked, np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * STEPS_PER_S, np.inf)
    back = np.r_[np.inf, step[:-1]]
    ok = seen & (back <= MAX_BALL_MPS) & (step <= MAX_BALL_MPS) & (speed <= MAX_BALL_MPS)
    return v1._rolling_max(np.where(ok, speed, np.nan), period, round(v1.KICK_S * STEPS_PER_S))


def _known_ball(t):
    """
    Ball position in metres, or the last seen one while it is hidden, within each half.
    """
    seen, period = t["ball_seen"], t["period"]
    xy = np.where(seen[:, None], t["ball"][:, :2].astype(float) * t["half_pitch"], np.nan)
    return np.stack([v1._carry_forward(xy[:, k], period) for k in (0, 1)], axis=1)


def keepers(t):
    """
    Keeper position per team and step, NaN when a team has nobody near its goal line.
    """
    half = t["half_pitch"].astype(float)
    out = np.full((len(t["period"]), 2, 2), np.nan)
    for k, (team, sign) in enumerate((("home", -1.0), ("away", 1.0))):
        xy = t[team][..., :2].astype(float) * half
        line = half[0] - sign * xy[..., 0]                    # metres to the team's own goal line
        ok = t[team + "_mask"] & (line < KEEPER_LINE_M) & (np.abs(xy[..., 1]) < KEEPER_HALF_WIDTH_M)
        line = np.where(ok, line, np.inf)
        if not line.shape[1]:
            continue
        best = np.argmin(line, axis=1)
        found = np.isfinite(line[np.arange(len(line)), best])
        out[found, k] = xy[np.flatnonzero(found), best[found]]
    return out


def keeper_ball_features(t):
    """
    ball_gk_dist and last_ball_in_box per step.
    """
    half = t["half_pitch"].astype(float)
    ball = _known_ball(t)
    with v1._quiet():
        gk = np.nanmin(np.linalg.norm(keepers(t) - ball[:, None], axis=-1), axis=1)
    with np.errstate(invalid="ignore"):
        box = ((np.abs(ball[:, 0]) > half[0] - v1.BOX_DEPTH_M) & (np.abs(ball[:, 0]) <= half[0])
               & (np.abs(ball[:, 1]) < v1.BOX_HALF_WIDTH_M))
    return np.stack([gk, np.where(np.isfinite(ball[:, 0]), box, np.nan)], axis=1)


def step_inputs(t, walking_speed, touchline_m, ball=True, collective=True, last_seen=True,
                formation=False, ball_exit=False, speed_cap=0.0, accel_cap=0.0, bump_s=0.0,
                bump_mps=0.0, whistle=False, tracked=False, keeper=False, feature_set="v2"):
    """
    The v2 inputs, columns in step_names() order.
    Raises if the [features] switches are not the ones v2 expects.
    """
    if not (ball and collective and last_seen) or formation or ball_exit or keeper \
            or speed_cap or accel_cap or bump_s or bump_mps:
        raise ValueError("feature_set v2 needs ball, collective and last_seen on, and formation, "
                         "ball_exit, keeper and the speed caps / bumps off")
    period = t["period"]
    robust = robust_signals(t)
    players = v1.step_features(t, walking_speed, touchline_m)
    players[:, PLAYER_NAMES.index("mean_speed")] = robust["mean_speed"]
    players[:, PLAYER_NAMES.index("speed_change")] = _central_rate(
        robust["median"], period, round(SPEED_CHANGE_S * STEPS_PER_S))
    balls = v1.ball_step_features(t)
    balls[:, v1.BALL_NAMES.index("ball_kick")] = robust_ball_kick(t)
    collective_x = v1.collective_step_features(t, robust)[:, _COLLECTIVE_KEEP]
    raw = [players, balls[:, _BALL_KEEP], collective_x, v1.last_seen_step_features(t),
           np.c_[robust["valid_frac"], robust["median"]], keeper_ball_features(t)]
    if whistle:
        raw.append(np.asarray(t.get("whistle", np.zeros(len(period))), np.float32)[:, None])
    if tracked:
        raw.append(v1.tracked_step_features(t))
    return np.column_stack(raw)


def step_names(whistle=False, tracked=False):
    """
    Column names matching step_inputs().
    """
    return NAMES + (v1.WHISTLE_NAMES if whistle else ()) + (v1.TRACKED_NAMES if tracked else ())
