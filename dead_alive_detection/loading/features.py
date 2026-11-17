"""
Per-step inputs of the TCN, not smoothed: the network learns its own time scales.

Feature groups: players, ball, collective stop, last-seen ball, formation, ball exit, referee
whistle, tracked-player counts and goalkeeper, each switched on in the config's [features].
Most counts are fractions of the players tracked, so DFL and BAR-ATH mean the same, and positions
are distances to lines and spots rather than raw x, y.
"""

import numpy as np

from dead_alive_detection.loading.tensors import STEPS_PER_S

# Laws of the game, not hyperparameters
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
NAMES = ("mean_speed", "walking_frac", "speed_change", "home_spread", "away_spread",
         "centroid_gap", "box_frac", "touchline_frac")
BALL_NAMES = ("ball_seen", "since_seen_s", "until_seen_s", "goal_line_dist", "touchline_dist",
              "corner_dist", "centre_dist", "ball_speed", "ball_kick", "still_s",
              "nearest_player", "near_ball_frac")
COLLECTIVE_NAMES = ("p90_speed", "running_frac", "speed_std", "stop_drop", "restart_rise",
                    "stopped_s", "stopped_ahead_s", "decel_frac")
LAST_SEEN_NAMES = ("last_line_dist", "last_toward_line")
FORMATION_NAMES = ("own_half_frac", "cluster_frac", "nn_dist", "centre_frac")
EXIT_NAMES = ("exit_eta_s", "projected_line_dist")
WHISTLE_NAMES = ("whistle",)
TRACKED_NAMES = ("home_tracked", "away_tracked")
KEEPER_NAMES = ("keeper_ball_dist", "keeper_speed", "keeper_goal_dist", "last_in_box")

# Params
STILL_MPS = 1.0       # a ball slower than this is still
KICK_S = 0.4          # ball_kick looks this far either side of a step
NEAR_BALL_M = 5.0     # a player closer than this is near the ball
RUN_MPS = 4.0         # a player faster than this is running
STOPPED_MPS = 2.0     # a p90_speed below this: everyone has stopped
DECEL_MPS = 1.0       # a player slowing down by more than this within 1 s is braking
CLUSTER_M = 4.0       # players closer than this to one player form its cluster
CENTRE_M = 15.0       # a player closer than this to the centre spot is near it
EXIT_MIN_MPS = 0.5    # slower than this towards the line: not heading out
EXIT_CAP_S = 10.0     # exit_eta_s when not heading out (and its largest value)
PROJECTED_FLOOR_M = 20.0
CHUNK = 2000          # steps per block of pairwise player distances (memory)
KEEPER_MAX_M = 25.0   # a team's player nearest its goal centre is its keeper only this close to it


def _masked_mean(values, mask):
    """
    Mean over the last axis where mask is True, NaN where it is all False.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return (values * mask).sum(-1) / mask.sum(-1)


def _centroid(xy, mask):
    """
    Mean position of the players present at each step.
    """
    return np.stack([_masked_mean(xy[..., k], mask) for k in (0, 1)], axis=-1)


def _spread(xy, mask):
    """
    Mean distance in metres of a team's players to their centroid.
    """
    distance = np.linalg.norm(xy - _centroid(xy, mask)[:, None], axis=-1)
    return _masked_mean(np.nan_to_num(distance), mask)


def _rate(x, period):
    """
    Per-second rate of change of a per-step series, within each half.
    """
    out = np.full(len(x), np.nan)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if len(rows) > 1:
            out[rows] = np.gradient(x[rows]) * STEPS_PER_S
    return out


def player_speeds(t, speed_cap=0.0, accel_cap=0.0):
    """
    Speed of every player slot per step, and a mask of the speeds to use.
    The optional caps drop implausible speeds and accelerations, which on BAR-ATH usually mean
    the slot switched to another player.
    """
    v = np.concatenate([t["home"][..., 2:], t["away"][..., 2:]], axis=1)
    mask = np.concatenate([t["home_mask"], t["away_mask"]], axis=1)
    speed = np.linalg.norm(v, axis=-1)
    use = mask.copy()
    if speed_cap > 0:
        use &= speed <= speed_cap
    if accel_cap > 0:
        period = t["period"]
        known = mask & (v != 0).any(axis=-1)          # tensors store an unknown velocity as 0
        ok = np.zeros(mask.shape, bool)
        ok[1:-1] = known[2:] & known[:-2] & (period[2:] == period[:-2])[:, None]
        accel = np.zeros(mask.shape)
        accel[1:-1] = np.linalg.norm((v[2:] - v[:-2]).astype(float), axis=-1) * STEPS_PER_S / 2
        use &= ~(ok & (accel > accel_cap))
    return speed, use


def _bumps(x, period, width_s, height, eps=0.3):
    """
    Steps inside a short bump of x: a rise and fall narrower than width_s and at least height tall.
    Same idea as bumps() in tracking_cleaning/cleaning.py, at 5 Hz.
    """
    from scipy.ndimage import grey_opening
    out = np.zeros(len(x), bool)
    size = max(3, int(round(width_s * STEPS_PER_S)) | 1)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        known = np.isfinite(x[rows])
        if known.sum() < 2:
            continue
        index = np.arange(len(rows))
        filled = np.interp(index, index[known], x[rows][known])
        excess = filled - grey_opening(filled, size=size)
        edges = np.flatnonzero(np.diff(np.r_[False, excess > eps, False]))
        for a, b in zip(edges[0::2], edges[1::2]):
            if (excess[a:b] > height).any():
                while a > 0 and 0 < excess[a - 1] < excess[a]:
                    a -= 1
                while b < len(excess) and 0 < excess[b] < excess[b - 1]:
                    b += 1
                out[rows[a:b]] = True
    return out


def _bridge(x, mask, period):
    """
    x with every run of mask replaced by a straight line between its neighbours, within each half.
    """
    out = x.astype(float).copy()
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for a, b in zip(edges[0::2], edges[1::2]):
        left = out[a - 1] if a > 0 and period[a - 1] == period[a] else np.nan
        right = out[b] if b < len(out) and period[b] == period[a] else np.nan
        left, right = (right if np.isnan(left) else left), (left if np.isnan(right) else right)
        if np.isfinite(left) and np.isfinite(right):
            out[a:b] = np.linspace(left, right, b - a + 2)[1:-1]
    return out


def speed_signals(t, walking_speed=1.0, speed_cap=0.0, accel_cap=0.0, bump_s=0.0, bump_mps=0.0):
    """
    Per-step team speed signals that every speed feature is built from.
    With bump_s and bump_mps set, short p90 bumps left by identity switches are flattened out.
    """
    period = t["period"]
    raw, use = player_speeds(t, speed_cap, accel_cap)
    speed = np.where(use, raw, np.nan)
    with np.errstate(invalid="ignore"), _quiet():
        p90 = np.nanpercentile(speed, 90, axis=1)
        spread = np.nanstd(speed, axis=1)
    second = round(STEPS_PER_S)                                         # 1 s in steps
    before = np.full_like(speed, np.nan)                                # each slot 1 s ago
    before[second:] = speed[:-second]
    crossed = np.zeros(len(period), bool)                              # 1 s ago was the other half
    crossed[second:] = period[second:] != period[:-second]
    before[crossed] = np.nan
    braking = use & np.isfinite(before)
    out = {"mean_speed": _masked_mean(raw, use),
           "walking_frac": _masked_mean(raw < walking_speed, use),
           "p90": p90,
           "running_frac": _masked_mean(raw > RUN_MPS, use),
           "spread": spread,
           "braking": _masked_mean(np.nan_to_num(before - speed) > DECEL_MPS, braking)}
    if bump_s > 0 and bump_mps > 0:
        bump = _bumps(p90, period, bump_s, bump_mps)
        out = {k: _bridge(v, bump, period) for k, v in out.items()}
    return out


def step_features(t, walking_speed, touchline_m, signals=None):
    """
    The eight player features per step, columns in NAMES order.
    """
    signals = speed_signals(t, walking_speed) if signals is None else signals
    half = t["half_pitch"]
    home, away = t["home"][..., :2] * half, t["away"][..., :2] * half       # back to metres
    players = np.concatenate([home, away], axis=1)
    mask = np.concatenate([t["home_mask"], t["away_mask"]], axis=1)
    mean_speed = signals["mean_speed"]
    in_box = ((np.abs(players[..., 0]) > half[0] - BOX_DEPTH_M)
              & (np.abs(players[..., 1]) < BOX_HALF_WIDTH_M))
    near_touchline = np.abs(players[..., 1]) > half[1] - touchline_m
    centroid_gap = np.linalg.norm(_centroid(home, t["home_mask"]) - _centroid(away, t["away_mask"]),
                                  axis=-1)
    return np.stack([mean_speed,
                     signals["walking_frac"],
                     _rate(mean_speed, t["period"]),
                     _spread(home, t["home_mask"]),
                     _spread(away, t["away_mask"]),
                     centroid_gap,
                     _masked_mean(in_box, mask),
                     _masked_mean(near_touchline, mask)], axis=1)


def _run_length(mask, period):
    """
    Per step, how many steps in a row mask has been True, counted within each half.
    """
    out = np.zeros(len(mask))
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        m = mask[rows]
        index = np.arange(len(m))
        last_false = np.maximum.accumulate(np.where(~m, index, -1))
        out[rows] = np.where(m, index - last_false, 0)
    return out


def _rolling_max(x, period, half_width):
    """
    Max of x over +/- half_width steps, within each half, ignoring NaN.
    """
    out = np.full(len(x), np.nan)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if not len(rows):
            continue
        padded = np.pad(np.where(np.isfinite(x[rows]), x[rows], -np.inf), half_width,
                        constant_values=-np.inf)
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_width + 1)
        best = windows.max(axis=1)
        out[rows] = np.where(np.isfinite(best), best, np.nan)
    return out


def ball_step_features(t):
    """
    The ball features per step, columns in BALL_NAMES order.
    """
    half, period, seen = t["half_pitch"], t["period"], t["ball_seen"]
    xy = np.where(seen[:, None], t["ball"][:, :2] * half, np.nan)          # metres
    to_lines = half - np.abs(xy)                                           # (goal line, touchline)

    # Velocity is a difference over +/-1 step (loading.tensors): unknown next to a gap.
    around = seen & np.r_[False, seen[:-1]] & np.r_[seen[1:], False]
    speed = np.where(around, np.linalg.norm(t["ball"][:, 2:], axis=-1), np.nan)
    with np.errstate(invalid="ignore"):
        still = speed < STILL_MPS

    players = np.concatenate([t["home"][..., :2], t["away"][..., :2]], axis=1) * half
    mask = np.concatenate([t["home_mask"], t["away_mask"]], axis=1)
    distance = np.where(mask, np.linalg.norm(players - xy[:, None], axis=-1), np.inf)
    nearest = distance.min(axis=1)

    return np.stack([seen.astype(float),
                     _run_length(~seen, period) / STEPS_PER_S,
                     _run_length(~seen[::-1], period[::-1])[::-1] / STEPS_PER_S,
                     to_lines[:, 0],
                     to_lines[:, 1],
                     np.linalg.norm(to_lines, axis=1),
                     np.linalg.norm(xy, axis=1),
                     speed,
                     _rolling_max(speed, period, round(KICK_S * STEPS_PER_S)),
                     np.where(seen, _run_length(still, period) / STEPS_PER_S, np.nan),
                     np.where(np.isfinite(nearest), nearest, np.nan),
                     np.where(seen, _masked_mean(distance < NEAR_BALL_M, mask), np.nan)], axis=1)


def shifted_mean(x, period, lo, hi):
    """
    Mean of x over steps i + lo to i + hi - 1 for every step i, within each half, ignoring NaN.
    Also used by augment/ball/fit_hidden.py.
    """
    out = np.full(len(x), np.nan)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        values = x[rows]
        known = np.isfinite(values)
        total = np.r_[0.0, np.cumsum(np.where(known, values, 0.0))]
        count = np.r_[0, np.cumsum(known)]
        a = np.clip(np.arange(len(rows)) + lo, 0, len(rows))
        b = np.clip(np.arange(len(rows)) + hi, 0, len(rows))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows] = (total[b] - total[a]) / (count[b] - count[a])
    return out


def collective_step_features(t, signals=None):
    """
    The collective-stop features per step, columns in COLLECTIVE_NAMES order.
    """
    signals = speed_signals(t) if signals is None else signals
    period = t["period"]
    p90 = signals["p90"]
    second = round(STEPS_PER_S)                                         # 1 s in steps
    now = shifted_mean(p90, period, 1 - second, 1)                     # the last 1 s
    with np.errstate(invalid="ignore"):
        stopped = p90 < STOPPED_MPS
    return np.stack([p90,
                     signals["running_frac"],
                     signals["spread"],
                     shifted_mean(p90, period, -3 * second, 1 - second) - now,
                     shifted_mean(p90, period, 1, 1 + second) - now,
                     _run_length(stopped, period) / STEPS_PER_S,
                     _run_length(stopped[::-1], period[::-1])[::-1] / STEPS_PER_S,
                     signals["braking"]], axis=1)


def _carry_forward(x, period):
    """
    Per step, the last finite value of x so far, within each half.
    """
    out = np.full(len(x), np.nan)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        values = x[rows]
        index = np.arange(len(rows))
        last = np.maximum.accumulate(np.where(np.isfinite(values), index, -1))
        out[rows] = np.where(last >= 0, values[np.maximum(last, 0)], np.nan)
    return out


def last_seen_step_features(t):
    """
    Where the ball was last seen relative to the lines, columns in LAST_SEEN_NAMES order.
    """
    half, period, seen = t["half_pitch"], t["period"], t["ball_seen"]
    xy = np.where(seen[:, None], t["ball"][:, :2] * half, np.nan)          # metres
    to_lines = half - np.abs(xy)                                           # (goal line, touchline)
    goal_line = to_lines[:, 0] < to_lines[:, 1]                            # which line is nearer

    # Outward speed: the velocity component pointing at the nearer line. Velocity is only known
    # with the ball seen on both neighbouring steps (as in ball_step_features).
    around = seen & np.r_[False, seen[:-1]] & np.r_[seen[1:], False]
    v = t["ball"][:, 2:]
    outward = np.where(goal_line, v[:, 0] * np.sign(xy[:, 0]), v[:, 1] * np.sign(xy[:, 1]))
    outward = np.where(around, outward, np.nan)

    return np.stack([_carry_forward(np.min(to_lines, axis=1), period),
                     _carry_forward(outward, period)], axis=1)


def formation_step_features(t):
    """
    The formation features per step, columns in FORMATION_NAMES order.
    """
    half = t["half_pitch"]
    home, away = t["home"][..., :2] * half, t["away"][..., :2] * half
    players = np.concatenate([home, away], axis=1)
    mask = np.concatenate([t["home_mask"], t["away_mask"]], axis=1)
    # Home always attacks +x (loading/tensors.py): its own half is x < 0, the away team's x > 0.
    own = np.concatenate([home[..., 0] < 0, away[..., 0] > 0], axis=1)
    n = mask.sum(axis=1)
    cluster = np.full(len(players), np.nan)
    nearest = np.full(len(players), np.nan)
    for a in range(0, len(players), CHUNK):
        xy, m = players[a:a + CHUNK], mask[a:a + CHUNK]
        d = np.linalg.norm(xy[:, :, None] - xy[:, None, :], axis=-1)
        pair = m[:, :, None] & m[:, None, :]
        close = ((d < CLUSTER_M) & pair).sum(axis=2)                   # includes the player itself
        with np.errstate(invalid="ignore", divide="ignore"):
            cluster[a:a + CHUNK] = np.where(n[a:a + CHUNK] > 0,
                                            close.max(axis=1) / n[a:a + CHUNK], np.nan)
        d = np.where(pair, d, np.inf)
        d[:, np.arange(d.shape[1]), np.arange(d.shape[1])] = np.inf     # not the player itself
        other = d.min(axis=2)
        with _quiet():
            nearest[a:a + CHUNK] = np.nanmedian(np.where(m & np.isfinite(other), other, np.nan), axis=1)
    centre = np.linalg.norm(players, axis=-1) < CENTRE_M
    return np.stack([_masked_mean(own, mask), cluster, nearest, _masked_mean(centre, mask)], axis=1)


def tracked_step_features(t):
    """
    Number of tracked players per team at each step.
    """
    return np.stack([t["home_mask"].sum(axis=1), t["away_mask"].sum(axis=1)], axis=1).astype(float)


def keepers(t):
    """
    Each team's keeper position and speed per step, taken as its player nearest its own goal.
    NaN when nobody is within KEEPER_MAX_M of the goal.
    """
    half = t["half_pitch"]
    out = np.full((len(t["period"]), 2, 3), np.nan)
    for k, (side, goal_x) in enumerate((("home", -half[0]), ("away", half[0]))):
        xy = t[side][..., :2] * half
        d = np.where(t[side + "_mask"], np.hypot(xy[..., 0] - goal_x, xy[..., 1]), np.inf)
        if d.shape[1] == 0:
            continue
        j = d.argmin(axis=1)
        rows = np.arange(len(j))
        ok = d[rows, j] < KEEPER_MAX_M
        out[ok, k, :2] = xy[rows, j][ok]
        out[ok, k, 2] = np.linalg.norm(t[side][rows, j, 2:4], axis=-1)[ok]
    return out


def keeper_step_features(t):
    """
    The goalkeeper features per step, columns in KEEPER_NAMES order.
    """
    half, period, seen = t["half_pitch"], t["period"], t["ball_seen"]
    ball = np.where(seen[:, None], t["ball"][:, :2] * half, np.nan)        # metres, seen only
    gk = keepers(t)
    side = (ball[:, 0] > 0).astype(int)                                    # 0: home defends, 1: away
    rows = np.arange(len(ball))
    with np.errstate(invalid="ignore"):
        near = np.where(seen, np.linalg.norm(ball - gk[rows, side, :2], axis=1), np.nan)
        in_box = ((half[0] - np.abs(ball[:, 0]) < BOX_DEPTH_M)
                  & (np.abs(ball[:, 1]) < BOX_HALF_WIDTH_M)).astype(float)
    in_box[~seen] = np.nan
    last_x = _carry_forward(ball[:, 0], period)
    defending = np.where(np.isfinite(last_x), (last_x > 0).astype(int), 0)
    keeper = gk[rows, defending]
    goal_dist = half[0] - np.abs(keeper[:, 0])
    out = np.stack([_carry_forward(near, period), keeper[:, 2], goal_dist,
                    _carry_forward(in_box, period)], axis=1)
    out[~np.isfinite(last_x), 1:3] = np.nan                                # no ball seen yet
    return out


def exit_step_features(t):
    """
    The ball-exit features per step, columns in EXIT_NAMES order.
    """
    period, seen = t["period"], t["ball_seen"]
    line_dist, outward = last_seen_step_features(t).T                  # carried through gaps
    since = _run_length(~seen, period) / STEPS_PER_S
    # Now, with the ball seen and its velocity known (outward is then this step's own value).
    around = seen & np.r_[False, seen[:-1]] & np.r_[seen[1:], False]
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.where(line_dist <= 0, 0.0,
                       np.where(outward > EXIT_MIN_MPS,
                                np.minimum(line_dist / outward, EXIT_CAP_S), EXIT_CAP_S))
        projected = np.maximum(line_dist - np.maximum(np.nan_to_num(outward), 0.0) * since,
                               -PROJECTED_FLOOR_M)
    return np.stack([np.where(around, eta, np.nan), projected], axis=1)


class _quiet:
    """
    Silences 'All-NaN slice' warnings, which are expected when a step has no players.
    """
    def __enter__(self):
        import warnings
        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)


def step_inputs(t, walking_speed, touchline_m, ball=False, collective=False, last_seen=False,
                formation=False, ball_exit=False, speed_cap=0.0, accel_cap=0.0, bump_s=0.0,
                bump_mps=0.0, whistle=False, tracked=False, keeper=False, feature_set="v1"):
    """
    All per-step TCN inputs, columns in step_names() order.
    The keyword arguments mirror a config's [features] section; feature_set="v2" uses features_v2.py.
    """
    if feature_set != "v1":
        from dead_alive_detection.loading import features_v2
        if feature_set != "v2":
            raise ValueError("unknown feature_set {!r}: v1 or v2".format(feature_set))
        return features_v2.step_inputs(t, walking_speed, touchline_m, ball, collective, last_seen,
                                       formation, ball_exit, speed_cap, accel_cap, bump_s,
                                       bump_mps, whistle, tracked, keeper)
    signals = speed_signals(t, walking_speed, speed_cap, accel_cap, bump_s, bump_mps)
    raw = [step_features(t, walking_speed, touchline_m, signals)]
    if ball:
        raw.append(ball_step_features(t))
    if collective:
        raw.append(collective_step_features(t, signals))
    if last_seen:
        raw.append(last_seen_step_features(t))
    if formation:
        raw.append(formation_step_features(t))
    if ball_exit:
        raw.append(exit_step_features(t))
    if whistle:
        raw.append(np.asarray(t.get("whistle", np.zeros(len(t["row"]))), np.float32)[:, None])
    if tracked:
        raw.append(tracked_step_features(t))
    if keeper:
        raw.append(keeper_step_features(t))
    return np.column_stack(raw)


def step_names(ball=False, collective=False, last_seen=False, formation=False, ball_exit=False,
               whistle=False, tracked=False, keeper=False, feature_set="v1", **_):
    """
    Column names matching step_inputs(). Also accepts a whole [features] section.
    """
    if feature_set == "v2":
        from dead_alive_detection.loading import features_v2
        return features_v2.step_names(whistle, tracked)
    return (NAMES + (BALL_NAMES if ball else ()) + (COLLECTIVE_NAMES if collective else ())
            + (LAST_SEEN_NAMES if last_seen else ()) + (FORMATION_NAMES if formation else ())
            + (EXIT_NAMES if ball_exit else ()) + (WHISTLE_NAMES if whistle else ())
            + (TRACKED_NAMES if tracked else ()) + (KEEPER_NAMES if keeper else ()))
