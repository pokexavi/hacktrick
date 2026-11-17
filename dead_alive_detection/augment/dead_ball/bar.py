"""
Make DFL's frozen dead ball move like BAR-ATH's ball in dead time, using real BAR pieces.

Builds a library of short ball movements from BAR-ATH stretches that the models vote dead, then
moves the DFL ball through each dead stretch by chaining those pieces. Labels, players and live
play are untouched.

    python -m dead_alive_detection.augment.dead_ball.bar
"""

import argparse
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from external import schema

# Params
VOTERS = ("tcn_players_only", "tcn_uniform_fragments", "tcn_uniform_learned")   # cache/dead_alive/<name>/frames.csv
MIN_VOTES = 2              # a BAR frame is dead when at least this many voters say so
MIN_STRETCH_S = 2.0        # shorter voted stretches are skipped
PIECE_S = {"start": 1.0, "middle": 3.0, "end": 1.0}   # library piece length per phase: long
                           # middle pieces keep BAR's still spells together (3 s matched BAR's
                           # middle-phase still share, 1 s pieces halved it); start and end last 2 s
PHASE_S = 2.0              # first / last this long of a stretch are its start / end phase
MAX_STEP_M = 1.2           # a frame-to-frame jump above this (30 m/s) is a tracker artefact
SCALE_RANGE = (0.75, 1.3)  # one speed scale per unfreeze() call
MARGIN_M = 3.0             # the ball stays within this of the pitch lines
STILL_MS = 0.3             # "still" below this speed (m/s), for the comparison only
SPEED_HALF_S = 0.5         # comparison speed: displacement over +/- this long
PHASES = ("start", "middle", "end")
BAR = os.path.join(schema.ROOT, "data", "processed", "BAR-ATH.parquet")
LIBRARY = os.path.join(schema.ROOT, "cache", "dead_alive_dfl", "bar_motion", "library.npz")
FPS = schema.FPS
LIBRARY_KEYS = ("voters", "min_votes", "min_stretch_s", "piece_s", "phase_s", "max_step_m")
DEFAULTS = {"voters": list(VOTERS), "min_votes": MIN_VOTES, "min_stretch_s": MIN_STRETCH_S,
            "piece_s": dict(PIECE_S), "phase_s": PHASE_S, "max_step_m": MAX_STEP_M}


def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def phase_of(k, n, phase_s=PHASE_S):
    """
    Phase (0 start, 1 middle, 2 end) of offsets k in a stretch of n frames.
    """
    edge = int(phase_s * FPS)
    k = np.asarray(k)
    return np.where(k < edge, 0, np.where(n - k <= edge, 2, 1))


def voted_dead(n_frames, folder=None, voters=VOTERS, min_votes=MIN_VOTES):
    """
    Frames that at least min_votes of the voters call dead on BAR.
    """
    folder = folder or os.path.join(schema.ROOT, "cache", "dead_alive")
    votes = np.zeros(n_frames, dtype=np.int8)
    for name in voters:
        f = pd.read_csv(os.path.join(folder, name, "frames.csv"), usecols=["frame", "state"])
        votes[f["frame"].to_numpy()] += (f["state"] == "D").to_numpy()
    return votes >= min_votes


def build_library(ball, period, dead, piece_s=PIECE_S, phase_s=PHASE_S,
                  min_stretch_s=MIN_STRETCH_S, max_step_m=MAX_STEP_M):
    """
    Ball displacement pieces from the dead stretches, grouped by phase.
    """
    xy = ball[:, :2].astype(float)
    out = {name: [] for name in PHASES}
    for a, b in _runs(dead):
        if b - a < min_stretch_s * FPS or period[a] != period[b - 1]:
            continue
        step = np.diff(xy[a:b], axis=0)                       # step k: frame a+k -> a+k+1
        ok = np.isfinite(step).all(axis=1) & (np.linalg.norm(step, axis=1) <= max_step_m)
        for p, name in enumerate(PHASES):
            piece = int(piece_s[name] * FPS)
            for c, d in _runs(ok):
                for s in range(c, d - piece + 1, piece):
                    if phase_of(s + piece // 2, b - a, phase_s) == p:
                        out[name].append(step[s:s + piece])
    return {name: np.asarray(v, dtype=np.float32).reshape(-1, int(piece_s[name] * FPS), 2)
            for name, v in out.items()}


def counts(library):
    return {name: len(library[name]) for name in PHASES}


def save_library(library, path=LIBRARY, params=None):
    """
    Save the library with the parameters it was built with.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **library, params=json.dumps(params or DEFAULTS, sort_keys=True))


def load_library(path=LIBRARY, params=None):
    """
    Load the library, refusing it if it was built with other parameters than params.
    """
    with np.load(path) as data:
        library = {k: data[k] for k in data.files if k != "params"}
        built = json.loads(str(data["params"])) if "params" in data.files else DEFAULTS
    if params is not None and json.dumps(built, sort_keys=True) != json.dumps(params, sort_keys=True):
        raise ValueError("{} was built with {}, the config asks for {}: delete it or point "
                         "[augment.dead_ball.bar] library elsewhere".format(path, built, params))
    return library


SIGNS = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)], dtype=float)


def _path_steps(n, library, rng, scale, target, phase_s=PHASE_S):
    """
    Displacements for a stretch of n frames made of library pieces, ending at target.
    Pieces are mirrored to stay near the straight line, and what is left is spread over the moving frames.
    """
    pools = [library[name] if len(library[name]) else library["middle"] for name in PHASES]
    out = np.empty((n, 2))
    k, here = 0, np.zeros(2)
    while k < n:
        pool = pools[int(phase_of(k, n, phase_s))]
        piece = pool[rng.integers(len(pool))].astype(float) * scale
        take = min(len(piece), n - k)
        piece = piece[:take]
        aim = target * (k + take) / n
        signs = SIGNS[np.argmin(np.linalg.norm(here + SIGNS * piece.sum(axis=0) - aim, axis=1))]
        out[k:k + take] = piece * signs
        here = here + out[k:k + take].sum(axis=0)
        k += take
    moved = np.linalg.norm(out, axis=1)
    share = moved / moved.sum() if moved.sum() > 0 else np.full(n, 1 / n)
    return out + share[:, None] * (target - here)


def _mirror(p, limit):
    p = np.where(p > limit, 2 * limit - p, p)
    p = np.where(p < -limit, -2 * limit - p, p)
    return np.clip(p, -limit, limit)


def unfreeze(match, label, rng, library, scale=None, scale_range=SCALE_RANGE, margin_m=MARGIN_M,
             phase_s=PHASE_S):
    """
    Copy of the match with the ball moving through every dead stretch of label, using BAR pieces.
    """
    if scale is None:
        scale = rng.uniform(*scale_range)
    ball = match.ball.copy()
    limit = np.asarray(match.pitch, dtype=float) / 2 + margin_m
    for a, b in _runs(label == "D"):
        if b >= len(ball) or b - a < 3 or not np.isfinite(ball[[a, b], :2]).all():
            continue
        steps = _path_steps(b - a, library, rng, scale, ball[b, :2] - ball[a, :2], phase_s)
        walk = np.r_[np.zeros((1, 2)), np.cumsum(steps, axis=0)]   # (n + 1, 2), walk[0] = 0
        ball[a:b, :2] = _mirror(ball[a, :2] + walk[:-1], limit)
    return replace(match, ball=ball, meta=dict(match.meta, ball_unfrozen="bar", motion_scale=scale))


def speed(ball, period):
    """
    Ball speed in m/s from the displacement over +/- SPEED_HALF_S, NaN across halves.
    """
    k = int(SPEED_HALF_S * FPS)
    xy = ball[:, :2].astype(float)
    v = np.full(len(xy), np.nan)
    v[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=1) / (2 * k / FPS)
    bad = np.ones(len(xy), dtype=bool)
    bad[k:-k] = period[2 * k:] != period[:-2 * k]
    v[bad] = np.nan
    return v


def profile(ball, period, dead):
    """
    Ball speed stats per phase over the dead stretches, for comparison.
    """
    v = speed(ball, period)
    rows = {p: [] for p in PHASES}
    still_runs = {p: [] for p in PHASES}
    for a, b in _runs(dead):
        if b - a < MIN_STRETCH_S * FPS:
            continue
        ph = phase_of(np.arange(b - a), b - a)
        for p in range(3):
            x = v[a:b][ph == p]
            rows[PHASES[p]].append(x[np.isfinite(x)])
            still_runs[PHASES[p]] += [(d - c) / FPS for c, d in _runs(np.isfinite(x) & (x < STILL_MS))]
    out = {}
    for p in PHASES:
        x = np.concatenate(rows[p]) if rows[p] else np.array([])
        out[p] = {"frames": len(x), "median": np.median(x) if len(x) else np.nan,
                  "p90": np.percentile(x, 90) if len(x) else np.nan,
                  "p99": np.percentile(x, 99) if len(x) else np.nan,
                  "still": (x < STILL_MS).mean() if len(x) else np.nan,
                  "still_run_s": np.median(still_runs[p]) if still_runs[p] else 0.0}
    return pd.DataFrame(out).T


def check():
    """
    Check on synthetic data that the library and the paths behave as expected.
    """
    rng = np.random.default_rng(0)
    n = 5000
    period = np.ones(n, dtype=np.int8)
    dead = np.zeros(n, dtype=bool)
    dead[1000:2000] = dead[3000:3500] = True
    xy = np.cumsum(rng.normal(0, 0.02, size=(n, 2)), axis=0)       # slow everywhere ...
    xy[1000:1050] += np.cumsum(np.full((50, 2), 0.1), axis=0)       # ... fast at a dead start
    xy[1000 + 50:2000] += 5.0
    ball = np.c_[xy, np.zeros(n)].astype(np.float32)
    ball[1500:1520] = np.nan                                       # a tracking gap
    lib = build_library(ball, period, dead)
    assert all(counts(lib).values()), counts(lib)
    start_speed = np.linalg.norm(lib["start"], axis=2).mean()
    middle_speed = np.linalg.norm(lib["middle"], axis=2).mean()
    assert start_speed > 2 * middle_speed, (start_speed, middle_speed)

    label = np.full(n, "H")
    label[200:900] = "D"
    frozen = np.zeros((n, 3), dtype=np.float32)
    frozen[900:, :2] = (10.0, -5.0)                                # restart spot
    match = schema.Match(name="synthetic", source="synthetic", frame=np.arange(n), period=period,
                         ball=frozen, home=np.zeros((n, 1, 2), np.float32),
                         away=np.zeros((n, 1, 2), np.float32), label=label,
                         meta={"pitch": [105.0, 68.0]})
    moved = unfreeze(match, label, np.random.default_rng(1), lib, scale=1.0)
    path = moved.ball[200:900, :2]
    assert np.allclose(path[0], frozen[200, :2]) and np.linalg.norm(path[-1] - frozen[900, :2]) < 0.5
    assert np.array_equal(moved.ball[:200], frozen[:200]) and np.array_equal(moved.ball[900:], frozen[900:])
    assert np.median(np.linalg.norm(np.diff(path, axis=0), axis=1)) > 0
    print("dead_ball.bar check passed: pieces {}".format(counts(lib)))


def library_params(section):
    """
    The library parameters from a config's [augment.dead_ball.bar] section.
    """
    return {k: (list(section[k]) if k == "voters" else dict(section[k]) if k == "piece_s"
                else section[k]) for k in LIBRARY_KEYS}


def build_from_bar(params, path, bar_path=BAR):
    """
    Build the library from BAR-ATH's voted dead stretches and save it.
    """
    bar = schema.load_match(bar_path)
    dead = voted_dead(len(bar.frame), voters=params["voters"], min_votes=params["min_votes"])
    library = build_library(bar.ball, bar.period, dead, params["piece_s"], params["phase_s"],
                            params["min_stretch_s"], params["max_step_m"])
    save_library(library, path, params)
    return library, bar, dead


def from_config(section, bar_path=BAR):
    """
    Unfreeze function from a config's [augment.dead_ball.bar] section.
    Builds the library first if its file is missing.
    """
    from dead_alive_detection.config import resolve_path
    params = library_params(section)
    path = resolve_path(section["library"])
    library = load_library(path, params) if os.path.exists(path) \
        else build_from_bar(params, path, bar_path)[0]
    return lambda match, label, rng: unfreeze(match, label, rng, library, None,
                                              tuple(section["scale_range"]), section["margin_m"],
                                              params["phase_s"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="synthetic test only")
    parser.add_argument("--config", help="take [augment.dead_ball.bar] from this config")
    args = parser.parse_args(argv)
    if args.check:
        check()
        return
    pd.set_option("display.width", 200)
    params, path = dict(DEFAULTS), LIBRARY
    if args.config:
        from dead_alive_detection import config
        section = config.load(args.config)["augment"]["dead_ball"]["bar"]
        params, path = library_params(section), config.resolve_path(section["library"])
    library, bar, dead = build_from_bar(params, path)
    print("library: pieces {} (lengths {} s), from {} voted dead stretches (>= {} of {})".format(
        counts(library), params["piece_s"], len(_runs(dead)), params["min_votes"],
        ", ".join(params["voters"])))
    print("\nBAR, voted dead, tracked frames:")
    print(profile(bar.ball, bar.period, dead).round(2).to_string())

    from dead_alive_detection.augment.dead_ball import uniform
    from dead_alive_detection.evaluation.metrics import truth_labels
    sims = {"frozen (DFL)": [], "ball_motion 2.7 m/s": [], "bar_motion": []}
    for i, path in enumerate(p for p in schema.processed_paths("dfl") if "J03WR9" not in p):
        m = schema.load_match(path)
        if m.source != "dfl":
            continue
        label = truth_labels(m)
        versions = {"frozen (DFL)": m,
                    "ball_motion 2.7 m/s": uniform.unfreeze(m, label, np.random.default_rng([8, i, 1])),
                    "bar_motion": unfreeze(m, label, np.random.default_rng([9, i, 1]), library)}
        for name, version in versions.items():
            sims[name].append((version.ball, version.period, label == "D"))
    for name, parts in sims.items():
        tables = [profile(*part) for part in parts]
        weights = pd.concat([t["frames"] for t in tables], axis=1)
        mean = sum(t.mul(t["frames"], axis=0) for t in tables).div(weights.sum(axis=1), axis=0)
        mean["frames"] = weights.sum(axis=1)
        print("\nDFL dead, {} (6 dev matches, frame-weighted):".format(name))
        print(mean.round(2).to_string())


if __name__ == "__main__":
    main()
