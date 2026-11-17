"""
Corrupt the players of a clean DFL match the way the BAR-ATH tracker does.

Each player is cut into pieces dropped into random anonymous slots, and duplicates are added
until the share of frames with more than 11 slots filled matches BAR. The numbers come from the
BAR noise profile. The ball is left alone.
"""

from dataclasses import replace
import json
import os

import numpy as np

from external import schema

# Params
PROFILE = os.path.join(schema.ROOT, "data", "processed", "noise_profile.json")
QUANTILES = np.linspace(0, 1, 41)       # the profile stores run lengths as 41 quantiles
DUPLICATE_MAX_S = 30.0                  # a duplicate lasts at most this long (assumption)
DUPLICATE_OFFSET_M = (0.3, 2.0)         # its fixed offset from the real player (assumption)
N_DRAWS = 20000                         # pre-drawn duplicate candidates per team
BISECTION_STEPS = 12


def load_profile(path=PROFILE):
    with open(path) as handle:
        return json.load(handle)


def sample(qs, size, rng):
    """
    Draw integers (at least 1) from a distribution stored as QUANTILES.
    """
    return np.maximum(1, np.round(np.interp(rng.random(size), QUANTILES, qs))).astype(np.int64)


def fragment(xy, period, team_prof, n_slots, rng):
    """
    Cut every player into tracklets and drop each into a random free slot.
    """
    n_frames, n_players = xy.shape[:2]
    real = np.isfinite(xy[..., 0])
    on_pitch = max(real[period > 0].sum(1).mean(), 1e-6)
    keep = min(team_prof["tracked_mean"] / on_pitch, 0.999)
    mean_piece = float(np.mean(sample(team_prof["tracklet_q"], 2000, rng)))
    mean_gap = mean_piece * (1 - keep) / keep

    pieces = []                                        # (start, stop, player)
    for k in range(n_players):
        t = int(rng.integers(0, max(int(mean_piece), 1)))
        while t < n_frames:
            length = int(sample(team_prof["tracklet_q"], 1, rng)[0])
            pieces.append((t, min(t + length, n_frames), k))
            t += length + int(rng.exponential(mean_gap)) + 1
    return assign_slots(xy, pieces, n_slots, rng)


def assign_slots(xy, pieces, n_slots, rng):
    """
    Drop each (start, stop, player) piece into a random free slot.
    """
    out = np.full((len(xy), n_slots, 2), np.nan, dtype=np.float32)
    free_at = np.zeros(n_slots, dtype=np.int64)
    for start, stop, k in sorted(pieces):
        free = np.flatnonzero(free_at <= start)
        if not len(free):
            continue
        slot = rng.choice(free)
        out[start:stop, slot] = xy[start:stop, k]
        free_at[slot] = stop
    out[~np.isfinite(out[..., 0])] = np.nan
    return out


def _duplicate_draws(n_players, team_prof, rng, max_s=DUPLICATE_MAX_S,
                     offset_m=DUPLICATE_OFFSET_M, n_draws=N_DRAWS):
    """
    All random draws for the duplicates, made once so the rate search is smooth.
    """
    cap = int(max_s * schema.FPS)
    angle = rng.uniform(0, 2 * np.pi, n_draws)
    radius = rng.uniform(*offset_m, n_draws)
    return {"gap": rng.exponential(1.0, n_draws),               # in units of 1 / rate
            "length": np.minimum(sample(team_prof["tracklet_q"], n_draws, rng), cap),
            "player": rng.integers(0, n_players, n_draws),
            "offset": np.stack([np.cos(angle), np.sin(angle)], 1) * radius[:, None],
            "slot_u": rng.random(n_draws)}


def _add_duplicates(clean, frag, draws, rate):
    """
    Add duplicate pieces at the given rate per frame, into slots free for their length.
    """
    out = frag.copy()
    n_frames = len(out)
    starts = np.cumsum(draws["gap"] / rate).astype(np.int64)
    for i in np.flatnonzero(starts < n_frames):
        start = starts[i]
        stop = min(start + draws["length"][i], n_frames)
        source = clean[start:stop, draws["player"][i]]
        if not np.isfinite(source[:, 0]).all():          # the player is not on the pitch
            continue
        free = np.flatnonzero(np.isnan(out[start:stop, :, 0]).all(0))
        if not len(free):
            continue
        slot = free[int(draws["slot_u"][i] * len(free))]
        out[start:stop, slot] = source + draws["offset"][i]
    return out


def _over_11(xy, period):
    counts = np.isfinite(xy[period > 0, :, 0]).sum(1)
    return float((counts > 11).mean())


def add_duplicates(clean, frag, period, target, team_prof, rng, bisection_steps=BISECTION_STEPS,
                   **draw_params):
    """
    Add duplicates at the rate that brings the share of frames with more than 11 slots close to target.
    """
    draws = _duplicate_draws(clean.shape[1], team_prof, rng, **draw_params)
    if target <= 0:
        return frag, 0.0
    low, high = 1e-6, 1e-2                               # duplicates per frame
    for _ in range(bisection_steps):                     # geometric bisection on the rate
        mid = np.sqrt(low * high)
        if _over_11(_add_duplicates(clean, frag, draws, mid), period) < target:
            low = mid
        else:
            high = mid
    rate = np.sqrt(low * high)
    return _add_duplicates(clean, frag, draws, rate), rate


def corrupt_players(match, profile, rng, duplicates=True, **params):
    """
    Copy of the match with BAR-like players. Ball, labels and frames are untouched.
    """
    teams = {}
    for side in ("home", "away"):
        prof = profile[side]
        clean = getattr(match, side)
        frag = fragment(clean, match.period, prof, prof["n_slots"], rng)
        if duplicates:
            frag, _ = add_duplicates(clean, frag, match.period, prof["over_11_frac"], prof, rng,
                                     **params)
        teams[side] = frag
    return replace(match, home=teams["home"], away=teams["away"],
                   meta=dict(match.meta, players_corrupted=True))


def summary(match):
    """
    Players tracked per team and share of frames with more than 11, to compare with BAR.
    """
    graded = match.period > 0
    out = {}
    for side in ("home", "away"):
        xy = getattr(match, side)
        counts = np.isfinite(xy[graded, :, 0]).sum(1)
        out[side] = {"slots": xy.shape[1], "tracked_mean": float(counts.mean()),
                     "over_11_frac": float((counts > 11).mean())}
    return out


def from_config(section):
    """
    Corrupt function (match, rng) -> match from a config's [augment.players.fragments] section.
    """
    from dead_alive_detection.config import resolve_path
    profile = load_profile(resolve_path(section["profile"]))
    params = {"bisection_steps": section["bisection_steps"], "max_s": section["duplicate_max_s"],
              "offset_m": tuple(section["duplicate_offset_m"]), "n_draws": section["n_draws"]}
    return lambda match, rng: corrupt_players(match, profile, rng, section["duplicates"], **params)
