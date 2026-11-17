"""
Turns 25 fps tracking into steps (5 per second): a token per player, one for the ball and one
for context, plus the training targets. No feature depends on who a slot is (in BAR it isn't a person).
"""

import numpy as np

FPS = 25
SCALE_POS, SCALE_V, SCALE_D = 50.0, 10.0, 100.0      # metres, m/s, metres: keeps values around [-1, 1]

# Player token, one per tracked player:
#   x, y                 where he is (home attacks +x)
#   vx, vy, ax, ay       velocity and acceleration: who runs towards the ball, who presses, who
#                        sprints into space after winning it
#   speed                how fast, whatever the direction
#   vel_ok               0 if the velocity can't be trusted (a gap, or the slot jumped to someone else)
#   home                 his team
#   d_plus, d_minus      distance to each goal: attacking or defending shape
#   ball_dx, ball_dy,    where the ball is from him: the one at the ball is usually in possession
#   ball_dist
#   ball_ok              0 when the ball isn't seen, so the three above mean nothing
PLAYER_COLS = ("x", "y", "vx", "vy", "ax", "ay", "speed", "vel_ok", "home", "d_plus", "d_minus",
               "ball_dx", "ball_dy", "ball_dist", "ball_ok")

# Ball token:
#   x, y, vx, vy         where it is and how it moves, when seen (a long fast pass vs a dribble)
#   seen                 BAR loses the ball ~41% of the time, the model has to know when
#   since                how long ago it was last seen: a fresh last position is still useful,
#                        an old one isn't
#   last_x, last_y       where it was last seen, so a hidden ball still has a rough position
#   d_plus, d_minus      distance to each goal (of the current or last seen ball)
#   has_last             0 if it hasn't been seen yet in this half
BALL_COLS = ("x", "y", "vx", "vy", "seen", "since", "last_x", "last_y", "d_plus", "d_minus", "has_last")

# Context token, one per step:
#   live                 live or dead step in the dead-ball mask. Dead steps stay in the window
#                        because how a restart is set up tells who restarts
#   since_start, to_end  time since the live stretch started and until it ends: right after a
#                        restart the restarting team has the ball
#   n_home, n_away       how many players each team has tracked: in BAR it varies a lot, so the
#                        model shouldn't read "few players" as "no possession"
CTX_COLS = ("live", "since_start", "to_end", "n_home", "n_away")
P = {c: i for i, c in enumerate(PLAYER_COLS)}
B = {c: i for i, c in enumerate(BALL_COLS)}
C = {c: i for i, c in enumerate(CTX_COLS)}
# Columns that flip sign when the pitch turns 180 degrees (away view) or is mirrored on the long axis
PLAYER_FLIP = [P[c] for c in ("x", "y", "vx", "vy", "ax", "ay", "ball_dx", "ball_dy")]
PLAYER_MIRROR = [P[c] for c in ("y", "vy", "ay", "ball_dy")]
BALL_FLIP = [B[c] for c in ("x", "y", "vx", "vy", "last_x", "last_y")]
BALL_MIRROR = [B[c] for c in ("y", "vy", "last_y")]


def step_rows(period, hz):
    # One frame every 25 / hz, starting at the first frame of each half
    stride = FPS // hz
    rows = [np.flatnonzero(period == p)[::stride] for p in (1, 2)]
    rows = np.concatenate(rows).astype(np.int64)
    return rows, period[rows].astype(np.int8)


def _shift(a, k):
    # a[t + k], NaN where it falls outside
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _smooth(xy, width):
    # Centred moving average, only where the whole window is tracked
    if width <= 1:
        return xy
    half = width // 2
    total = np.zeros_like(xy)
    count = np.zeros(xy.shape[:-1], dtype=np.int32)
    for k in range(-half, half + 1):
        part = _shift(xy, k)
        ok = np.isfinite(part[..., 0])
        total += np.where(ok[..., None], part, 0)
        count += ok
    full = count == 2 * half + 1
    return np.where(full[..., None], total / np.maximum(count, 1)[..., None], xy)


def kinematics(xy, s):
    # Velocity, whether to trust it, and acceleration. Too fast means the slot jumped to someone
    # else, so that velocity is marked bad and zeroed.
    sm = _smooth(xy, s["smooth_frames"])
    k, k2 = s["vel_frames"], s["acc_frames"]
    v = (_shift(sm, k) - _shift(sm, -k)) * FPS / (2 * k)
    v_ok = np.isfinite(v[..., 0]) & np.isfinite(xy[..., 0])
    speed = np.linalg.norm(np.nan_to_num(v), axis=-1)
    v_ok &= speed <= s["speed_cap"]
    v = np.where(v_ok[..., None], v, np.nan)
    a = (_shift(v, k2) - _shift(v, -k2)) * FPS / (2 * k2)
    a_ok = np.isfinite(a[..., 0]) & v_ok & (np.linalg.norm(np.nan_to_num(a), axis=-1) <= s["accel_cap"])
    return np.where(v_ok[..., None], v, 0.0), v_ok, np.where(a_ok[..., None], a, 0.0)


def _team_tokens(xy, v, v_ok, a, home, ball, ball_seen, half_length, max_n):
    # Tokens, mask and how many are tracked, for one team. Extra slots past max_n are dropped.
    valid = np.isfinite(xy[..., 0])
    t, s = valid.shape
    f = np.zeros((t, s, len(PLAYER_COLS)), dtype=np.float32)
    pos = np.nan_to_num(xy)
    f[..., P["x"]] = pos[..., 0] / SCALE_POS
    f[..., P["y"]] = pos[..., 1] / SCALE_POS
    f[..., [P["vx"], P["vy"]]] = v / SCALE_V
    f[..., [P["ax"], P["ay"]]] = a / SCALE_V
    f[..., P["speed"]] = np.linalg.norm(v, axis=-1) / SCALE_V
    f[..., P["vel_ok"]] = v_ok
    f[..., P["home"]] = float(home)
    f[..., P["d_plus"]] = np.hypot(pos[..., 0] - half_length, pos[..., 1]) / SCALE_D
    f[..., P["d_minus"]] = np.hypot(pos[..., 0] + half_length, pos[..., 1]) / SCALE_D
    rel = np.nan_to_num(ball[:, None, :2] - pos)
    f[..., P["ball_dx"]] = np.where(ball_seen[:, None], rel[..., 0], 0) / SCALE_POS
    f[..., P["ball_dy"]] = np.where(ball_seen[:, None], rel[..., 1], 0) / SCALE_POS
    f[..., P["ball_dist"]] = np.where(ball_seen[:, None], np.hypot(rel[..., 0], rel[..., 1]), 0) / SCALE_POS
    f[..., P["ball_ok"]] = ball_seen[:, None].astype(np.float32)
    f[~valid] = 0.0
    order = np.argsort(~valid, axis=1, kind="stable")[:, :max_n]
    tokens = np.take_along_axis(f, order[..., None], axis=1)
    mask = np.take_along_axis(valid, order, axis=1)
    if tokens.shape[1] < max_n:                         # not enough slots: pad with zeros
        pad = max_n - tokens.shape[1]
        tokens = np.concatenate([tokens, np.zeros((t, pad, f.shape[2]), np.float32)], axis=1)
        mask = np.concatenate([mask, np.zeros((t, pad), bool)], axis=1)
    return tokens, mask, valid.sum(1)


def _ball(ball_frames, rows, period, s, half_length, hz):
    # When the ball isn't seen, the token says where and how long ago it was last seen
    xy = ball_frames[:, :2].astype(np.float32)
    v, v_ok, _ = kinematics(xy, dict(s, speed_cap=40.0, accel_cap=1e9))
    pos = xy[rows]
    seen = np.isfinite(pos[:, 0])
    out = np.zeros((len(rows), len(BALL_COLS)), dtype=np.float32)
    out[:, B["x"]] = np.where(seen, pos[:, 0], 0) / SCALE_POS
    out[:, B["y"]] = np.where(seen, pos[:, 1], 0) / SCALE_POS
    out[:, [B["vx"], B["vy"]]] = np.where((seen & v_ok[rows])[:, None], v[rows], 0) / SCALE_V
    out[:, B["seen"]] = seen
    last = np.full((len(rows), 2), np.nan, dtype=np.float32)
    since = np.full(len(rows), s["since_cap_s"], dtype=np.float32)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        seen_at = np.where(seen[idx], np.arange(len(idx)), -1)
        latest = np.maximum.accumulate(seen_at)
        has = latest >= 0
        last[idx[has]] = pos[idx[latest[has]]]
        since[idx[has]] = (np.arange(len(idx))[has] - latest[has]) / hz
    has_last = np.isfinite(last[:, 0])
    out[:, B["since"]] = np.minimum(since, s["since_cap_s"]) / s["since_cap_s"]
    out[:, B["last_x"]] = np.where(has_last, last[:, 0], 0) / SCALE_POS
    out[:, B["last_y"]] = np.where(has_last, last[:, 1], 0) / SCALE_POS
    ref = np.where(seen[:, None], np.nan_to_num(pos), np.nan_to_num(last))
    out[:, B["d_plus"]] = np.where(seen | has_last, np.hypot(ref[:, 0] - half_length, ref[:, 1]), 0) / SCALE_D
    out[:, B["d_minus"]] = np.where(seen | has_last, np.hypot(ref[:, 0] + half_length, ref[:, 1]), 0) / SCALE_D
    out[:, B["has_last"]] = has_last
    return out, seen, np.nan_to_num(pos)


def _live_timing(live, period, hz, cap_s=30.0):
    # Time since the live stretch started and until it ends, capped at 30 s
    since, to_end = np.zeros(len(live), np.float32), np.zeros(len(live), np.float32)
    change = np.flatnonzero((live[1:] != live[:-1]) | (period[1:] != period[:-1])) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, len(live)]):
        if live[a]:
            n = b - a
            since[a:b] = np.minimum(np.arange(n) / hz, cap_s) / cap_s
            to_end[a:b] = np.minimum((n - 1 - np.arange(n)) / hz, cap_s) / cap_s
    return since, to_end


def targets(label, rows, period, hz, sigma_s):
    # y: 1 = H, 0 = A, NaN when dead. change: a bump around each change of possession,
    # exp(-dt^2 / 2 sigma^2) with dt the time to the closest one.
    lab = label[rows]
    y = np.where(lab == "H", 1.0, np.where(lab == "A", 0.0, np.nan)).astype(np.float32)
    f = np.arange(1, len(label))
    switch = np.isin(label[f], ("H", "A")) & np.isin(label[f - 1], ("H", "A")) & (label[f] != label[f - 1])
    nearest = np.full(len(rows), np.inf, np.float32)      # seconds to the closest change
    for p in (1, 2):
        steps = np.flatnonzero(period == p)
        if not len(steps):
            continue
        at = rows[steps]
        times = f[switch & (f >= at[0]) & (f <= at[-1])]   # first frame of the new team
        if not len(times):
            continue
        k = np.searchsorted(times, at)
        before = times[np.clip(k - 1, 0, len(times) - 1)]
        after = times[np.clip(k, 0, len(times) - 1)]
        nearest[steps] = np.minimum(np.abs(at - before), np.abs(at - after)) / FPS
    change = (np.exp(-0.5 * (nearest / sigma_s) ** 2) if sigma_s > 0
              else (nearest == 0).astype(np.float32)).astype(np.float32)
    return y, change


def build(cfg, version, live, target=None):
    # Everything the model reads for one match, plus the targets if we have labels (not for BAR)
    s, hz = cfg["steps"], cfg["steps"]["hz"]
    rows, period = step_rows(version.period, hz)
    half_length = version.pitch[0] / 2
    ball, seen, ball_pos = _ball(version.ball, rows, period, s, half_length, hz)
    teams = []
    for xy, home in ((version.home, True), (version.away, False)):
        v, v_ok, a = kinematics(xy.astype(np.float32), s)
        teams.append(_team_tokens(xy[rows], v[rows], v_ok[rows], a[rows], home,
                                  np.where(seen[:, None], ball_pos, np.nan), seen, half_length,
                                  s["max_per_team"]))
    step_live = live[rows] & (period > 0)
    since, to_end = _live_timing(step_live, period, hz)
    ctx = np.stack([step_live.astype(np.float32), since, to_end,
                    teams[0][2] / 11.0, teams[1][2] / 11.0], axis=1).astype(np.float32)
    item = {"players": np.concatenate([teams[0][0], teams[1][0]], axis=1).astype(np.float16),
            "pmask": np.concatenate([teams[0][1], teams[1][1]], axis=1),
            "ball": ball, "ctx": ctx, "rows": rows, "period": period, "live": step_live,
            "live_frames": live & (version.period > 0)}
    if target is not None:
        y, change = targets(target, rows, period, hz, cfg["heads"]["change_sigma_s"])
        item["y"], item["change"] = y, change
    return item
