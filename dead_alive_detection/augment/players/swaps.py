"""
Swap a clean DFL match's players between tracks the way BAR-ATH's tracker does.

Swaps are drawn from those fit_swaps.py found on BAR-ATH. A swapped track glides from the old
player to the new one, then follows the new one until the end of the half. Apply before the
player losses. Ball, labels and frames are untouched.
"""

from dataclasses import replace
import json

import numpy as np

from external import schema

# Params
DEAD_SHARE = 0.5         # share of the swaps put on dead-ball frames
ALIVE = ("H", "A")


def load_model(path):
    with open(path) as handle:
        return json.load(handle)


def _all_players(match):
    return np.concatenate([match.home, match.away], axis=1)


def _half_end(period, f):
    """
    Frame index just after the half that frame f is in.
    """
    same = np.flatnonzero(period == period[f])
    return int(same[-1]) + 1


def _nearest(xy, pool, who, target):
    """
    The player in pool whose distance to who is closest to target metres.
    """
    d = np.linalg.norm(xy[pool] - xy[who], axis=1)
    return int(pool[np.argmin(np.abs(d - target))])


def _members(xy, team, swap, rng):
    """
    Players and their targets for one swap at a frame, or None if it does not fit.
    """
    present = np.flatnonzero(np.isfinite(xy[:, 0]))
    if len(present) < 2:
        return None
    first = int(rng.choice(present))
    pool = present[present != first]
    k = int(swap["n_slots"])
    if k == 1:
        other = _nearest(xy, pool, first, swap["off_path_m"])
        return [first], [other]
    chosen = [first]
    target = swap["spread_m"] if np.isfinite(swap["spread_m"]) else 0.0
    for n in range(k - 1):
        if swap["cross_team"] and n == 0:
            candidates = pool[team[pool] != team[first]]   # at least one of the other team
        elif swap["cross_team"]:
            candidates = pool
        else:
            candidates = pool[team[pool] == team[first]]
        if not len(candidates):
            return None
        chosen.append(_nearest(xy, candidates, first, target))
        pool = pool[pool != chosen[-1]]
    return chosen, chosen[1:] + chosen[:1]


def plan(match, model, rng, dead_share=DEAD_SHARE, rate_scale=1.0):
    """
    The swaps to apply, sorted by start frame.
    """
    halves = np.isin(match.period, (1, 2))
    library = model["swaps"]
    if not library or not halves.any():
        return []
    minutes = halves.sum() / schema.FPS / 60
    n = rng.poisson(model["per_min"] * rate_scale * minutes)
    dead_frames = np.flatnonzero(halves & (match.label == "D"))
    alive_frames = np.flatnonzero(halves & np.isin(match.label, ALIVE))
    n_dead = rng.binomial(n, dead_share) if len(dead_frames) else 0
    if not len(alive_frames):
        n_dead = n if len(dead_frames) else 0
    xy = _all_players(match)
    team = np.r_[np.zeros(match.home.shape[1], int), np.ones(match.away.shape[1], int)]
    out = []
    for dead in [True] * n_dead + [False] * (n - n_dead):
        frames = dead_frames if dead else alive_frames
        if not len(frames):
            continue
        swap = library[int(rng.integers(len(library)))]
        start = int(rng.choice(frames))
        length = max(1, int(round(swap["duration_s"] * schema.FPS)))
        length = min(length, _half_end(match.period, start) - start)
        who = _members(xy[start], team, swap, rng)
        if who is None:
            continue
        out.append({"start": start, "frames": length, "players": who[0], "targets": who[1],
                    "dead": dead, "cross_team": bool(len({team[p] for p in who[0] + who[1]}) > 1)})
    return sorted(out, key=lambda s: s["start"])


def _smoothstep(n):
    """
    Weights from near 0 to near 1: the share of the new player at each frame.
    """
    u = (np.arange(n) + 1) / (n + 1)
    return u * u * (3 - 2 * u)


def apply(match, swaps):
    """
    Copy of the match with the planned swaps applied.
    """
    xy = _all_players(match)
    n_frames, n_players = xy.shape[:2]
    source = np.tile(np.arange(n_players, dtype=np.int16), (n_frames, 1))   # whom each track follows
    glides = []
    for s in swaps:
        a, b = s["start"], s["start"] + s["frames"]
        end = _half_end(match.period, a)
        players, targets = s["players"], s["targets"]
        before = source[a:b, players].copy()
        source[a:end, players] = source[a:end, targets]
        glides.append((a, b, players, before, source[a:b, players].copy()))
    rows = np.arange(n_frames)[:, None]
    out = xy[rows, source]
    for a, b, players, before, after in glides:
        w = _smoothstep(b - a)[:, None, None]
        frames = np.arange(a, b)[:, None]
        old, new = xy[frames, before], xy[frames, after]
        mixed = (1 - w) * old + w * new
        mixed = np.where(np.isnan(old), new, np.where(np.isnan(new), old, mixed))
        out[a:b, players] = mixed
    n_home = match.home.shape[1]
    return replace(match, home=out[:, :n_home].astype(match.home.dtype),
                   away=out[:, n_home:].astype(match.away.dtype))


def describe(match, swaps):
    """
    One readable row per swap, for looking at them.
    """
    n_home = match.home.shape[1]
    name = lambda p: "home {}".format(p) if p < n_home else "away {}".format(p - n_home)
    rows = []
    for s in swaps:
        period = int(match.period[s["start"]])
        first = int(np.flatnonzero(match.period == period)[0])
        rows.append({"period": period, "start_s": round((s["start"] - first) / schema.FPS, 1),
                     "seconds": round(s["frames"] / schema.FPS, 1), "n_players": len(s["players"]),
                     "players": ", ".join(name(p) for p in s["players"]),
                     "cross_team": s["cross_team"], "dead": s["dead"],
                     "columns": sorted(set(s["players"]) | set(s["targets"]))})
    return rows


def corrupt_swaps(match, model, rng, dead_share=DEAD_SHARE, rate_scale=1.0):
    """
    Copy of the match with players swapped as on BAR-ATH. meta["swaps"] lists the swaps.
    """
    swaps = plan(match, model, rng, dead_share, rate_scale)
    out = apply(match, swaps)
    return replace(out, meta=dict(match.meta, swaps=describe(match, swaps)))


def from_config(section):
    """
    Corrupt function (match, rng) -> match from a config's [augment.swaps] section.
    """
    from dead_alive_detection.config import resolve_path
    model = load_model(resolve_path(section["model"]))
    return lambda match, rng: corrupt_swaps(match, model, rng, section["dead_share"],
                                            section["rate_scale"])
