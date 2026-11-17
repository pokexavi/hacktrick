"""
Hide the ball in a DFL match the way the BAR-ATH tracker loses it.

Runs the lose and find models from fit_hidden.py step by step. The lose bias is shifted until
the share of hidden steps reaches a target, by default the one measured on BAR-ATH.
"""

import json
import math
from dataclasses import replace

import numpy as np

from dead_alive_detection.augment.ball.fit_hidden import PLAYERS, context
from dead_alive_detection.loading.tensors import STEPS_PER_S, match_tensors, to_frames

# Params
SHIFT_RANGE = (-8.0, 8.0)     # search range of the lose-bias shift, in log-odds
BISECTION_STEPS = 14          # 16 / 2**14: the shift is found to ~0.001


def load_model(path):
    with open(path) as handle:
        return json.load(handle)


def _score(part, ctx, names):
    """
    Weighted sum of the standardised features, without the bias. Undefined features add 0.
    """
    total = np.zeros(len(ctx["players_speed"]))
    for name in names:
        k = part["features"].index(name)
        z = (ctx[name] - part["mean"][k]) / part["std"][k]
        total += part["weights"][k] * np.nan_to_num(z)
    return total


def _simulate(rows, lose, lose_no_speed, find_now, find_lost, hidden_weight, u, shift):
    """
    Hidden mask of one half, starting with the ball visible.
    """
    hidden = np.zeros(len(rows), dtype=bool)
    visible, was_visible, last = True, True, rows[0]
    for k, i in enumerate(rows):
        hidden[k] = not visible
        if visible:
            logit = (lose[i] if was_visible else lose_no_speed[i]) + shift
            last = i
            changes = u[i] < 1 / (1 + math.exp(-logit))
        else:
            logit = find_now[i] + find_lost[last] + hidden_weight((i - last) / STEPS_PER_S)
            changes = u[i] < 1 / (1 + math.exp(-logit))
        was_visible = visible
        visible = visible != changes
    return hidden


def hidden_steps(t, model, rng, hidden_frac=None):
    """
    Mask of the hidden steps and the lose-bias shift that reaches the target share.
    hidden_frac defaults to the share measured on BAR-ATH.
    """
    target = model["hidden_frac"] if hidden_frac is None else hidden_frac
    if target >= 1:
        return np.ones(len(t["period"]), dtype=bool), np.inf
    if target <= 0:                     # the clean ball: a stoppage seen with every frame of it
        return np.zeros(len(t["period"]), dtype=bool), -np.inf
    ctx = context(t, t["ball_seen"])
    lose, find = model["lose"], model["find"]
    lose_ball = [name for name in lose["features"] if name not in PLAYERS]
    find_ball = [name for name in find["features"] if name not in PLAYERS + ("hidden_s",)]

    lose_score = lose["bias"] + _score(lose, ctx, lose_ball) + _score(lose, ctx, list(PLAYERS))
    lose_no_speed = lose_score - _score(lose, ctx, ["ball_speed"])
    find_now = find["bias"] + _score(find, ctx, list(PLAYERS))
    find_lost = _score(find, ctx, find_ball)
    h = find["features"].index("hidden_s")
    hidden_weight = lambda s: find["weights"][h] * (s - find["mean"][h]) / find["std"][h]  # noqa: E731

    u = rng.random(len(t["period"]))
    halves = [np.flatnonzero(t["period"] == p) for p in (1, 2)]
    halves = [rows for rows in halves if len(rows)]

    def run(shift):
        hidden = np.zeros(len(u), dtype=bool)
        for rows in halves:
            hidden[rows] = _simulate(rows, lose_score, lose_no_speed, find_now, find_lost,
                                     hidden_weight, u, shift)
        return hidden

    low, high = SHIFT_RANGE
    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2
        if run(mid).mean() < target:
            low = mid
        else:
            high = mid
    shift = (low + high) / 2
    return run(shift), shift


def hide_ball(t, model, rng, hidden_frac=None):
    """
    Copy of the tensors with the ball hidden by the model. This is what training uses.
    """
    hidden, _ = hidden_steps(t, model, rng, hidden_frac)
    out = dict(t)
    out["ball_seen"] = t["ball_seen"] & ~hidden
    out["ball"] = np.where(hidden[:, None], 0, t["ball"]).astype(t["ball"].dtype)
    return out


def corrupt_match(match, model, rng, hidden_frac=None):
    """
    Copy of the match with the ball hidden like on BAR-ATH, plus a report of aggregates.
    Labels are only read for the report.
    """
    t = match_tensors(match)
    hidden, shift = hidden_steps(t, model, rng, hidden_frac)
    frames = to_frames(hidden, t["row"], len(match.frame)) & (match.period > 0)
    ball = match.ball.copy()
    ball[frames] = np.nan

    # Steps where the ball really disappears: hidden here, or never tracked in the source.
    lost = hidden | ~t["ball_seen"]
    players = context(t, t["ball_seen"])["players_speed"]
    report = {"shift": shift, "hidden_frac": float(lost.mean()),
              "gap_quartiles_s": _gap_quartiles(lost, t["period"]),
              "players_speed_hidden": float(np.nanmedian(players[lost])),
              "players_speed_visible": float(np.nanmedian(players[~lost])),
              "hidden_in_dead": float(lost[t["dead"] & t["scored"]].mean()),
              "hidden_in_alive": float(lost[~t["dead"] & t["scored"]].mean())}
    return replace(match, ball=ball), report


def _gap_quartiles(hidden, period):
    """
    Quartiles in seconds of the hidden run lengths, split at halftime.
    """
    change = np.flatnonzero((hidden[1:] != hidden[:-1]) | (period[1:] != period[:-1])) + 1
    lengths = [(b - a) / STEPS_PER_S for a, b in zip(np.r_[0, change], np.r_[change, len(hidden)])
               if hidden[a]]
    return np.percentile(lengths, [25, 50, 75]).round(1).tolist() if lengths else []
