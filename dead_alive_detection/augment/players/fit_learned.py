"""
Learn from BAR-ATH when its tracker loses a player and when it finds them again.

Same idea as ball/fit_hidden.py, per player. BAR-ATH has no identities, so losses are tracklets
ending and finds come from stitching an ended tracklet to a nearby new one. The models are saved
to cache/players_learned/ for players/learned.py.

    python -m dead_alive_detection.augment.players.fit_learned --match <BAR-ATH .parquet>
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from dead_alive_detection.augment.ball.fit_hidden import CROWD_M, PLAYERS, context
from dead_alive_detection.augment.players.fragments import QUANTILES
from dead_alive_detection.loading.clean import played
from dead_alive_detection.loading.tensors import STEP, STEPS_PER_S, match_tensors
from external import schema

# Params
JUMP_M = 4.0                 # a slot moving further than this in one step (20 m/s) changed person
MAX_GAP_S = 60.0             # a hidden stretch longer than this is not stitched
REACH_M = 3.0                # stitch: distance allowed whatever the gap (tracker error)
REACH_MPS = 8.0              # stitch: distance allowed per second of gap
CHUNK = 2000                 # steps per batch when counting the crowd: the pairs are heavy
TEAM_SIZE = 11
PLACE = ("crowd", "goal_dist", "touch_dist", "centroid_dx", "centroid_dy")
LOSE = ("own_speed", "log_visible_s") + PLACE + PLAYERS
FIND = ("log_hidden_s",) + PLACE + PLAYERS
OUT = os.path.join(schema.ROOT, "cache", "players_learned")


def positions(t):
    """
    Positions in metres and presence of both teams' slots, home first.
    """
    xy = np.concatenate([t["home"][..., :2], t["away"][..., :2]], axis=1) * t["half_pitch"]
    return xy, np.concatenate([t["home_mask"], t["away_mask"]], axis=1)


def own_speed(xy, mask, period):
    """
    Each player's speed since the previous step, NaN if they were not there.
    """
    out = np.full(mask.shape, np.nan)
    moved = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * STEPS_PER_S
    both = mask[1:] & mask[:-1] & (period[1:] == period[:-1])[:, None]
    out[1:] = np.where(both, moved, np.nan)
    return out


def place_context(xy, mask, half):
    """
    Where each present player is (the PLACE features), NaN for empty slots.
    """
    crowd = np.zeros(mask.shape)
    for a in range(0, len(xy), CHUNK):
        part, present = xy[a:a + CHUNK], mask[a:a + CHUNK]
        dist = np.linalg.norm(part[:, :, None] - part[:, None], axis=-1)
        close = (dist < CROWD_M) & present[:, :, None] & present[:, None]
        crowd[a:a + CHUNK] = close.sum(axis=2) - 1                     # not counting themselves
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    centroid = (xy * mask[..., None]).sum(axis=1, keepdims=True) / count[..., None]
    out = {"crowd": crowd,
           "goal_dist": half[0] - np.abs(xy[..., 0]),
           "touch_dist": half[1] - np.abs(xy[..., 1]),
           "centroid_dx": np.abs(xy[..., 0] - centroid[..., 0]),
           "centroid_dy": np.abs(xy[..., 1] - centroid[..., 1])}
    return {name: np.where(mask, values, np.nan) for name, values in out.items()}


def collective(t):
    """
    The players' speed features around each step, as in ball/fit_hidden.
    """
    ctx = context(t, t["ball_seen"])
    return {name: ctx[name] for name in PLAYERS}


def tracklets(xy, mask, period, jump_m=JUMP_M):
    """
    (start, stop, slot) of every tracklet, sorted by start.
    A tracklet ends at halftime, when the slot empties, or when it jumps further than jump_m in one step.
    """
    out = []
    breaks = np.r_[True, period[1:] != period[:-1]]
    step = np.zeros(mask.shape, dtype=bool)
    step[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) > jump_m
    for slot in range(mask.shape[1]):
        present = mask[:, slot]
        starts = present & (breaks | step[:, slot] | np.r_[True, ~present[:-1]])
        begin = np.flatnonzero(starts)
        # A tracklet stops at the next start or at the next step it is absent.
        absent = np.r_[np.flatnonzero(~present), len(present)]
        end = absent[np.searchsorted(absent, begin)]
        stop = np.minimum(end, np.r_[begin[1:], len(present)])
        out += [(int(a), int(b), slot) for a, b in zip(begin, stop)]
    return sorted(out)


def stitch(tracks, xy, period, max_gap_s=MAX_GAP_S, reach_m=REACH_M, reach_mps=REACH_MPS):
    """
    Pairs (i, j) where tracklet j continues tracklet i after a gap.
    Each ended tracklet is linked at most once, to the new one it reaches most easily.
    """
    pairs = []
    waiting = []                                       # ended tracklets not linked yet, by stop
    ended = sorted(range(len(tracks)), key=lambda i: tracks[i][1])
    next_end = 0
    for j, (start, _, slot) in enumerate(tracks):
        while next_end < len(ended) and tracks[ended[next_end]][1] < start:
            waiting.append(ended[next_end])
            next_end += 1
        waiting = [i for i in waiting if (start - tracks[i][1]) / STEPS_PER_S <= max_gap_s
                   and period[tracks[i][1] - 1] == period[start]]
        best, best_ratio = None, 1.0
        for i in waiting:
            gap_s = (start - tracks[i][1]) / STEPS_PER_S
            dist = np.linalg.norm(xy[start, slot] - xy[tracks[i][1] - 1, tracks[i][2]])
            ratio = dist / (reach_m + reach_mps * gap_s)
            if ratio <= best_ratio:
                best, best_ratio = i, ratio
        if best is not None:
            pairs.append((best, j))
            waiting.remove(best)
    return pairs


def lose_rows(tracks, period, offset=0):
    """
    Rows for the lose model: one per tracked step with a next step in the same half.
    """
    has_next = np.r_[period[1:] == period[:-1], False]
    steps, players, ages, lost = [], [], [], []
    for start, stop, slot in tracks:
        rows = np.arange(start, stop)
        rows = rows[has_next[rows]]
        steps.append(rows)
        players.append(np.full(len(rows), slot + offset))
        ages.append(rows - start)
        lost.append(rows == stop - 1)
    return tuple(np.concatenate(x) for x in (steps, players, ages, lost))


def find_rows(tracks, pairs, offset=0):
    """
    Rows for the find model: one per hidden step of every stitched gap.
    """
    steps, players, away, last, found = [], [], [], [], []
    for i, j in pairs:
        gone = tracks[i][1] - 1                        # the last step they were seen on
        rows = np.arange(tracks[i][1], tracks[j][0])
        steps.append(rows)
        players.append(np.full(len(rows), tracks[i][2] + offset))
        away.append(rows - gone)
        last.append(np.full(len(rows), gone))
        found.append(rows == tracks[j][0] - 1)
    if not steps:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty, np.zeros(0, dtype=bool)
    return tuple(np.concatenate(x) for x in (steps, players, away, last, found))


def fit_part(x, outcome, names):
    """
    Fit a logistic regression on standardised features and return it as plain numbers.
    Undefined features are set to the mean.
    """
    mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    std = np.where(std > 0, std, 1.0)
    model = LogisticRegression(C=1.0)
    model.fit(np.nan_to_num((x - mean) / std), outcome)
    return {"features": list(names), "mean": mean.tolist(), "std": std.tolist(),
            "weights": model.coef_[0].tolist(), "bias": float(model.intercept_[0])}


def team_aggregates(mask, tracks, period_ok):
    """
    Per-team numbers the simulator needs: slots, players tracked, share over eleven, tracklet lengths.
    """
    counts = mask[period_ok].sum(axis=1)
    lengths = np.array([b - a for a, b, _ in tracks], dtype=float) * STEP
    return {"n_slots": int(mask.shape[1]), "tracked_mean": float(counts.mean()),
            "tracked_capped": float(np.minimum(counts, TEAM_SIZE).mean()),
            "over_11_frac": float((counts > TEAM_SIZE).mean()),
            "tracklet_q": np.quantile(lengths, QUANTILES).tolist()}


def fit(t):
    """
    Fit both models and the per-team aggregates from the tensors, and return them with a report.
    """
    period = t["period"]
    xy, mask = positions(t)
    place = place_context(xy, mask, t["half_pitch"])
    speed = own_speed(xy, mask, period)
    players = collective(t)
    n_home = t["home_mask"].shape[1]
    sides = (("home", 0, range(n_home)), ("away", n_home, range(n_home, mask.shape[1])))

    lose, find, aggregates, report = [], [], {}, {}
    for side, offset, slots in sides:
        team_xy, team_mask = xy[:, list(slots)], mask[:, list(slots)]
        tracks = tracklets(team_xy, team_mask, period)
        pairs = stitch(tracks, team_xy, period)
        lose.append(lose_rows(tracks, period, offset))
        find.append(find_rows(tracks, pairs, offset))
        aggregates[side] = team_aggregates(team_mask, tracks, np.ones(len(period), dtype=bool))
        report[side] = {"tracklets": len(tracks), "stitched": len(pairs)}

    # A tracklet's first step has no past: its speed is undefined there, as after a reappearance.
    steps, who, ages, lost = (np.concatenate(x) for x in zip(*lose))
    speed_now = np.where(ages > 0, speed[steps, who], np.nan)
    lose_x = {"own_speed": speed_now, "log_visible_s": np.log1p(ages / STEPS_PER_S)}
    lose_x.update({name: place[name][steps, who] for name in PLACE})
    lose_x.update({name: players[name][steps] for name in PLAYERS})
    models = {"lose": fit_part(np.column_stack([lose_x[n] for n in LOSE]), lost, LOSE)}

    steps, who, away, last, found = (np.concatenate(x) for x in zip(*find))
    find_x = {"log_hidden_s": np.log1p(away / STEPS_PER_S)}
    find_x.update({name: place[name][last, who] for name in PLACE})
    find_x.update({name: players[name][steps] for name in PLAYERS})
    models["find"] = fit_part(np.column_stack([find_x[n] for n in FIND]), found, FIND)
    models.update(aggregates)
    report["lose_rows"], report["losses"] = len(lost), int(lost.sum())
    report["find_rows"], report["finds"] = len(found), int(found.sum())
    return models, report


def print_part(name, part):
    """
    Print the weights as odds ratios per standard deviation.
    """
    print("\n{}: odds x per +1 SD".format(name))
    order = np.argsort(-np.abs(part["weights"]))
    for k in order:
        print("    {:<16} {:>8.2f}".format(part["features"][k], np.exp(part["weights"][k])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--match", required=True, help="processed match parquet, always explicit")
    args = parser.parse_args(argv)

    match = played(schema.load_match(args.match))
    models, report = fit(match_tensors(match))
    for side in ("home", "away"):
        a = models[side]
        print("{}: {} slots, {:.1f} tracked, > 11 on {:.1%} of steps, tracklet median {:.1f} s; "
              "{} tracklets, {} stitched".format(
                  side, a["n_slots"], a["tracked_mean"], a["over_11_frac"],
                  np.interp(0.5, QUANTILES, a["tracklet_q"]) / schema.FPS,
                  report[side]["tracklets"], report[side]["stitched"]))
    print_part("lose ({} events in {} steps)".format(report["losses"], report["lose_rows"]),
               models["lose"])
    print_part("find ({} events in {} steps)".format(report["finds"], report["find_rows"]),
               models["find"])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, match.name + ".json")
    with open(path, "w") as handle:
        json.dump(models, handle, indent=2)
    print("\nmodels saved to", path)


if __name__ == "__main__":
    main()
