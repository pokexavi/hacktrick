"""
Learn from BAR-ATH when its tracker loses the ball and when it finds it again.

Two logistic regressions over the 5 Hz steps, one for losing the ball and one for finding it.
The context uses only the players and the last seen ball, so it can be computed the same way on
DFL. No labels are read. The models are saved to cache/ball_loss/ for ball/hidden.py.

    python -m dead_alive_detection.augment.ball.fit_hidden --match <BAR-ATH .parquet>
"""

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dead_alive_detection.loading.features import (BOX_DEPTH_M, BOX_HALF_WIDTH_M, keepers,
                                                   shifted_mean)
from dead_alive_detection.loading.tensors import STEPS_PER_S, match_tensors
from external import schema

# Params
CROWD_M = 3.0                # players closer than this to the ball crowd it (occlusion)
KEEPER_HOLD_M = 2.0          # the defending keeper this close to the ball in his box: he has it
WINDOWS_S = (3, 10)          # players' mean speed over this long before and after a step
PLAYERS = ("players_speed",) + tuple("speed_{}_{}s".format(side, w)
                                     for side in ("past", "future") for w in WINDOWS_S)
LOSE = ("ball_speed", "goal_dist", "touch_dist", "outside", "crowd") + PLAYERS
FIND = ("hidden_s", "goal_dist", "touch_dist", "outside") + PLAYERS
KEEPER = ("keeper_holds",)   # added to both with --keeper
OUT = os.path.join(schema.ROOT, "cache", "ball_loss")


def _last_seen(seen):
    """
    Index of the last step at or before each step where the ball was seen, -1 if none.
    """
    return np.maximum.accumulate(np.where(seen, np.arange(len(seen)), -1))


def context(t, seen):
    """
    Every context feature as {name: (N,) array}, NaN where undefined.
    seen is passed apart from t so the simulator can use its own mask.
    """
    half = t["half_pitch"]
    period = t["period"]
    out = {}

    # Players: median speed (a slot that jumps to another person would dominate a mean).
    players = np.concatenate([t["home"], t["away"]], axis=1)
    mask = np.concatenate([t["home_mask"], t["away_mask"]], axis=1)
    speed = np.where(mask, np.linalg.norm(players[..., 2:], axis=-1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a step with no players is NaN
        out["players_speed"] = np.nanmedian(speed, axis=1)
    # How the players' speed changes around the step: mean over the window minus now. Past > 0:
    # they have slowed down. Future < 0: they are about to slow down; > 0: about to restart.
    now = out["players_speed"]
    for w in WINDOWS_S:
        k = round(w * STEPS_PER_S)
        past = shifted_mean(now, period, -k, 0)
        future = shifted_mean(now, period, 1, k + 1)
        out["speed_past_{}s".format(w)] = past - now
        out["speed_future_{}s".format(w)] = future - now

    # The last seen ball, per half: index of the step it was seen at, and its position in metres.
    last = np.full(len(seen), -1)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        local = _last_seen(seen[rows])
        last[rows] = np.where(local >= 0, rows[np.maximum(local, 0)], -1)
    known = last >= 0
    xy = np.where(known[:, None], t["ball"][np.maximum(last, 0), :2] * half, np.nan)
    out["goal_dist"] = half[0] - np.abs(xy[:, 0])          # negative behind a goal line
    out["touch_dist"] = half[1] - np.abs(xy[:, 1])         # negative beyond a touchline
    with np.errstate(invalid="ignore"):
        out["outside"] = np.where(known, (out["goal_dist"] < 0) | (out["touch_dist"] < 0), np.nan)
    out["hidden_s"] = np.where(known, (np.arange(len(seen)) - last) / STEPS_PER_S, np.nan)

    # Ball speed from the displacement since the previous step: it needs only the past, so it is
    # defined on the very step before a loss. NaN if the previous step did not see the ball.
    step_xy = t["ball"][:, :2] * half
    moved = np.linalg.norm(step_xy[1:] - step_xy[:-1], axis=1) * STEPS_PER_S
    both = seen[1:] & seen[:-1] & (period[1:] == period[:-1])
    out["ball_speed"] = np.r_[np.nan, np.where(both, moved, np.nan)]

    # Players crowding the last seen ball, counted at the current step.
    dist = np.linalg.norm(players[..., :2] * half - xy[:, None], axis=-1)
    out["crowd"] = np.where(known, ((dist < CROWD_M) & mask).sum(axis=1), np.nan)

    # The defending keeper (the player nearest the goal of the half the ball is in) next to the
    # last seen ball inside his penalty area: a save or a catch, where BAR-ATH may lose the ball
    # while play goes on. 0 where no keeper is found.
    gk = keepers(t)
    with np.errstate(invalid="ignore"):
        side = (xy[:, 0] > 0).astype(int)
        near = np.linalg.norm(xy - gk[np.arange(len(seen)), side, :2], axis=1) < KEEPER_HOLD_M
        in_box = (out["goal_dist"] >= 0) & (out["goal_dist"] < BOX_DEPTH_M) & \
            (np.abs(xy[:, 1]) < BOX_HALF_WIDTH_M)
    out["keeper_holds"] = np.where(known, (near & in_box).astype(float), np.nan)
    return out


def transitions(seen, period):
    """
    Rows and outcomes of the lose and find models: steps whose next step is in the same half.
    """
    nxt = np.r_[period[1:] == period[:-1], False]
    lose_rows = np.flatnonzero(nxt & seen)
    find_rows = np.flatnonzero(nxt & ~seen)
    return lose_rows, ~seen[lose_rows + 1], find_rows, seen[find_rows + 1]


def fit_one(ctx, names, rows, outcome):
    """
    Fit a standardised logistic regression on the rows where every feature is defined.
    """
    x = np.column_stack([ctx[name][rows] for name in names])
    keep = np.isfinite(x).all(axis=1)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
    model.fit(x[keep], outcome[keep])
    return model, int(keep.sum()), int(outcome[keep].sum())


def to_json(model, names):
    """
    The fitted model as plain numbers, so it can be applied without sklearn.
    """
    scaler, logistic = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return {"features": list(names), "mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(),
            "weights": logistic.coef_[0].tolist(), "bias": float(logistic.intercept_[0])}


def gap_lengths(seen, period):
    """
    Lengths in seconds of the gaps that neither start nor end a half.
    """
    lengths = []
    for p in (1, 2):
        hidden = ~seen[period == p]
        edges = np.diff(np.r_[0, hidden.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if a > 0 and b < len(hidden):
                lengths.append((b - a) / STEPS_PER_S)
    return np.array(lengths)


def report(name, model, names, n_rows, n_events):
    """
    Print the weights as odds ratios per standard deviation.
    """
    scaler, logistic = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    print("\n{}: {} steps, {} events ({:.3f} per second)".format(
        name, n_rows, n_events, n_events / n_rows * STEPS_PER_S))
    print("    {:<14} {:>10} {:>12}".format("feature", "1 SD", "odds x / SD"))
    order = np.argsort(-np.abs(logistic.coef_[0]))
    for k in order:
        print("    {:<14} {:>10.2f} {:>12.2f}".format(
            names[k], scaler.scale_[k], np.exp(logistic.coef_[0][k])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--match", required=True, help="processed match parquet, always explicit")
    parser.add_argument("--keeper", action="store_true", help="add keeper_holds to both models")
    args = parser.parse_args(argv)
    extra = KEEPER if args.keeper else ()

    match = schema.load_match(args.match)
    t = match_tensors(match)
    seen = t["ball_seen"]
    ctx = context(t, seen)
    lose_rows, lost, find_rows, found = transitions(seen, t["period"])

    lengths = gap_lengths(seen, t["period"])
    print("ball hidden on {:.1%} of steps; {} complete gaps, length quartiles {} s".format(
        1 - seen.mean(), len(lengths), np.round(np.percentile(lengths, [25, 50, 75]), 1).tolist()
        if len(lengths) else []))

    models = {}
    print("keeper_holds on {:.2%} of the steps with the ball seen".format(
        np.nanmean(ctx["keeper_holds"][seen])))
    for name, names, rows, outcome in (("lose", LOSE + extra, lose_rows, lost),
                                       ("find", FIND + extra, find_rows, found)):
        model, n_rows, n_events = fit_one(ctx, names, rows, outcome)
        report(name, model, names, n_rows, n_events)
        models[name] = to_json(model, names)

    models["hidden_frac"] = float(1 - seen.mean())
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, match.name + ("_keeper" if args.keeper else "") + ".json")
    with open(path, "w") as handle:
        json.dump(models, handle, indent=2)
    print("\nmodels saved to", path)


if __name__ == "__main__":
    main()
