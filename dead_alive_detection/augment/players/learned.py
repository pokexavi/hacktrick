"""
Make a clean DFL match's players vanish and come back the way the BAR-ATH tracker loses them.

Runs the lose and find models from fit_learned.py for each player, then puts the visible pieces
into anonymous slots and adds duplicates as players/fragments.py does. The find bias is shifted
per team until the number of tracked players matches BAR.
"""

from dataclasses import replace
import json

import numpy as np

from dead_alive_detection.augment.players import fit_learned
from dead_alive_detection.augment.players.fragments import add_duplicates, assign_slots
from dead_alive_detection.loading.tensors import STEP, STEPS_PER_S, match_tensors

# Params
SHIFT_RANGE = (-8.0, 8.0)     # search range of the find-bias shift, in log-odds
BISECTION_STEPS = 8           # 16 / 2**8: the shift is found to ~0.06
SIDES = ("home", "away")


def load_model(path):
    with open(path) as handle:
        return json.load(handle)


def _score(part, ctx, names, shape):
    """
    Weighted sum of the standardised features, without the bias. Undefined features add 0.
    """
    total = np.zeros(shape)
    for name in names:
        k = part["features"].index(name)
        z = (ctx[name] - part["mean"][k]) / part["std"][k]
        total = total + part["weights"][k] * np.nan_to_num(z)
    return total


def _time_table(part, name, n):
    """
    A model's time term (weight times standardised log1p seconds) for 0 to n - 1 steps.
    """
    k = part["features"].index(name)
    z = (np.log1p(np.arange(n) / STEPS_PER_S) - part["mean"][k]) / part["std"][k]
    return part["weights"][k] * z


def _simulate(parts, real, u, period, shift):
    """
    Which players are visible on each step, for a given find-bias shift per player.
    """
    lose, lose_no_speed, find_now, find_lost, seen_term, away_term = parts
    n_steps, n_players = real.shape
    players = np.arange(n_players)
    visible_out = np.zeros(real.shape, dtype=bool)
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows):
            continue
        visible = real[rows[0]].copy()
        was = np.zeros(n_players, dtype=bool)          # visible on the previous step too
        age = np.zeros(n_players, dtype=np.int64)      # steps visible so far
        last = np.full(n_players, rows[0])             # the last step they were seen on
        for i, j in zip(rows[:-1], rows[1:]):
            visible_out[i] = visible
            last = np.where(visible, i, last)
            logit = np.where(
                visible,
                np.where(was, lose[i], lose_no_speed[i]) + seen_term[age],
                find_now[i] + find_lost[last, players] + away_term[i - last] + shift)
            changes = u[i] < 1 / (1 + np.exp(-logit))
            coming = visible != changes
            coming = (coming | (real[j] & ~real[i])) & real[j]    # substitutes come on seen
            was = visible & coming
            age = np.where(was, age + 1, 0)
            visible = coming
        visible_out[rows[-1]] = visible
    return visible_out


def visibility(t, model, rng, bisection_steps=BISECTION_STEPS, crowd_scale=1.0):
    """
    Visible mask of both teams' players, home first, and each team's find shift.
    crowd_scale > 1 loses more players in crowds than the fit says.
    """
    xy, real = fit_learned.positions(t)
    period = t["period"]
    lose, find = model["lose"], model["find"]
    ctx = fit_learned.place_context(xy, real, t["half_pitch"])
    ctx.update({name: values[:, None] for name, values in fit_learned.collective(t).items()})
    ctx["own_speed"] = fit_learned.own_speed(xy, real, period)
    shape = real.shape

    place = fit_learned.PLACE
    collective = fit_learned.PLAYERS
    lose_no_speed = lose["bias"] + _score(lose, ctx, place + collective, shape)
    if crowd_scale != 1.0:
        lose_no_speed = lose_no_speed + (crowd_scale - 1.0) * _score(lose, ctx, ("crowd",), shape)
    parts = (lose_no_speed + _score(lose, ctx, ("own_speed",), shape), lose_no_speed,
             find["bias"] + _score(find, ctx, collective, shape), _score(find, ctx, place, shape),
             _time_table(lose, "log_visible_s", len(period) + 1),
             _time_table(find, "log_hidden_s", len(period) + 1))

    n_home = t["home_mask"].shape[1]
    team = np.r_[np.zeros(n_home, dtype=np.int64), np.ones(shape[1] - n_home, dtype=np.int64)]
    # BAR's share of a full team, on this match's players: a sent-off player is not missed.
    on_pitch = np.array([real[:, team == k].sum(axis=1).mean() for k in (0, 1)])
    target = on_pitch * [model[s]["tracked_capped"] / fit_learned.TEAM_SIZE for s in SIDES]

    u = rng.random(shape)

    def counts(shift):
        visible = _simulate(parts, real, u, period, shift[team])
        return visible, np.array([visible[:, team == k].sum(axis=1).mean() for k in (0, 1)])

    low, high = np.full(2, SHIFT_RANGE[0]), np.full(2, SHIFT_RANGE[1])
    for _ in range(bisection_steps):
        mid = (low + high) / 2
        short = counts(mid)[1] < target
        low, high = np.where(short, mid, low), np.where(short, high, mid)
    shift = (low + high) / 2
    return counts(shift)[0], shift


def _pieces(visible, rows, period, n_frames):
    """
    (start, stop, player) frames of every run of visible steps of one team.
    """
    pieces = []
    for half in (1, 2):
        steps = np.flatnonzero(period == half)
        if not len(steps):
            continue
        end = min(int(rows[steps[-1]]) + STEP, n_frames)     # a half's last step lasts STEP frames
        for player in range(visible.shape[1]):
            edges = np.diff(np.r_[0, visible[steps, player].astype(np.int8), 0])
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                stop = end if b == len(steps) else int(rows[steps[b]])
                pieces.append((int(rows[steps[a]]), stop, player))
    return pieces


def corrupt_players(match, model, rng, duplicates=True, bisection_steps=BISECTION_STEPS,
                    duplicate_bisection_steps=12, crowd_scale=1.0, **params):
    """
    Copy of the match with players lost as on BAR-ATH. Ball, labels and frames are untouched.
    """
    t = match_tensors(match)
    visible, _ = visibility(t, model, rng, bisection_steps, crowd_scale)
    n_home = t["home_mask"].shape[1]
    teams = {}
    for side, part in zip(SIDES, (visible[:, :n_home], visible[:, n_home:])):
        prof = model[side]
        clean = getattr(match, side)
        pieces = _pieces(part, t["row"], t["period"], len(match.frame))
        slots = assign_slots(clean, pieces, prof["n_slots"], rng)
        if duplicates:
            slots, _ = add_duplicates(clean, slots, match.period, prof["over_11_frac"], prof, rng,
                                      duplicate_bisection_steps, **params)
        teams[side] = slots
    return replace(match, home=teams["home"], away=teams["away"],
                   meta=dict(match.meta, players_corrupted="learned"))


def from_config(section):
    """
    Corrupt function (match, rng) -> match from a config's [augment.players.learned] section.
    """
    from dead_alive_detection.config import resolve_path
    model = load_model(resolve_path(section["model"]))
    params = {"max_s": section["duplicate_max_s"], "offset_m": tuple(section["duplicate_offset_m"]),
              "n_draws": section["n_draws"]}
    return lambda match, rng: corrupt_players(match, model, rng, section["duplicates"],
                                              section["bisection_steps"],
                                              section["duplicate_bisection_steps"],
                                              section["crowd_scale"], **params)
