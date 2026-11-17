"""Shared loading, caching and per-frame signals for the exploration notebooks.

Everything is indexed by VIDEO frame (= submission row). Alignment comes from games.py and the
starter conventions: ATD video_frame = frame_id - 1, Smart Tagging video_frame = round(s * 25),
homography row = video frame.

Where the data comes from
-------------------------
games.DATA_DIR, i.e. ./data unless MTAH_DATA_DIR is set. To run any notebook on the synthetic
full-length bundle instead, set it BEFORE importing this module:

    os.environ["MTAH_DATA_DIR"] = os.path.join(ROOT, "cache", "synthetic_full")

Parsed data is cached as parquet under cache/parsed/<data-dir name>/<source>/, so the real and
synthetic caches never mix. A cache is rebuilt automatically when its source file changes (size or
mtime), or on demand with refresh=True. cache/ is gitignored: it is derived match data.
"""

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = games.GAMES[GAME]["pitch"]        # metres (105.3 x 68.0)
N_FRAMES = games.source_frames("tactical")                   # 199319
HALVES = games.half_video_blocks(GAME)                        # [(1, 11185, 90651), (2, 110808, 190197)]
SECOND_HALF_KICKOFF = 113742                                  # DATA.md / FAQ.md
PUBLIC_WINDOW = (11185, 18684)                                # inclusive, video frames
GT_PUBLIC_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BASELINE_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_ROOT = os.path.join(ROOT, "cache", "parsed")
OUTPUTS = os.path.join(ROOT, "outputs")
# Bump when the code that builds a cached table changes, so stale caches are rebuilt.
CACHE_VERSIONS = {"atd": 2, "signals": 2}

LABELS = ("H", "A", "D", "X")
LABEL_COLOURS = {"H": "#d62728", "A": "#1f77b4", "D": "#9e9e9e", "X": "#ffffff"}
TEAM_COLOURS = ("#d62728", "#1f77b4")                         # home, away -- as starter/utils.py
HOME, AWAY, NOBODY = 0, 1, -1

# Smart Tagging club names -> side. DATA.md spells the away club 'Atletic Club' in the tagging.
ST_TEAM_SIDE = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}


# -- where the data is ---------------------------------------------------------------------------

def data_dir():
    """The bundle in use: MTAH_DATA_DIR if set, else ./data. Read at call time, not import time."""
    return os.environ.get("MTAH_DATA_DIR") or games.DATA_DIR


def paths(source="tactical"):
    return games.asset_paths(data_dir=data_dir(), source=source)


def is_synthetic():
    """True when the bundle in use was written by exploration/synthetic/make_synthetic.py."""
    meta = paths()["atd_meta"]
    if not os.path.isfile(meta):
        return False
    import xml.etree.ElementTree as ET
    name = ET.parse(meta).getroot().findtext("Metadata/GlobalConfig/FileName") or ""
    return name.startswith("SYNTHETIC")


def describe_source():
    """One line saying which bundle the notebook is reading. Prints no data."""
    print("data dir: {}   synthetic: {}".format(data_dir(), is_synthetic()))


def _cache_dir(source):
    tag = os.path.basename(os.path.normpath(data_dir()))
    folder = os.path.join(CACHE_ROOT, tag, source)
    os.makedirs(folder, exist_ok=True)
    return folder


def _stamp(path):
    stat = os.stat(path)
    return {"path": os.path.abspath(path), "size": stat.st_size, "mtime": stat.st_mtime}


def _cached(name, source, inputs, build, refresh=False):
    """Load cache/<...>/<name>.parquet if it was built from exactly these input files, else build."""
    folder = _cache_dir(source)
    target = os.path.join(folder, name + ".parquet")
    stamp_path = os.path.join(folder, name + ".stamp.json")
    stamp = [{"version": CACHE_VERSIONS.get(name, 0)}] + [_stamp(p) for p in inputs]
    if not refresh and os.path.isfile(target) and os.path.isfile(stamp_path):
        with open(stamp_path) as handle:
            if json.load(handle) == stamp:
                return pd.read_parquet(target)
    table = build()
    table.to_parquet(target, index=False)
    with open(stamp_path, "w") as handle:
        json.dump(stamp, handle)
    return table


# -- tracking (ATD) -------------------------------------------------------------------------------

def load_meta(source="tactical"):
    """What the ATD metadata says: teams, the slot -> team table, channel units, periods."""
    import xml.etree.ElementTree as ET
    root = ET.parse(games.require(["atd_meta"], data_dir=data_dir(), source=source)[0]).getroot()
    meta = root.find("Metadata")

    def provider_params(node):
        return {p.findtext("Name"): p.findtext("Value") for p in node.iter("ProviderParameter")}

    teams = list(meta.iter("Team"))
    home_id = teams[0].get("id")      # FIFA/EPTS: first team is home (as starter/baseline.py)
    slots = list(meta.iter("Player"))
    return {
        "team_ids": [t.get("id") for t in teams],
        "team_names": [t.findtext("Name") for t in teams],
        "attack_direction_first_half": [provider_params(t).get("attack_direction_first_half")
                                        for t in teams],
        "slot_ids": [p.get("id") for p in slots],
        "slot_team": np.array([HOME if p.get("teamId") == home_id else AWAY for p in slots]),
        "channel_units": {c.get("id"): c.findtext("Unit") for c in meta.iter("Channel")},
        "global_params": provider_params(meta.find("GlobalConfig")),
        "field_size": (meta.findtext("Sessions/Session/FieldSize/Width"),
                       meta.findtext("Sessions/Session/FieldSize/Height")),
    }


def _parse_atd_raw(path):
    """The raw text as one wide table: frame, ball_x/y/s, s1_x/y/s ... (float32).

    Every separator becomes a comma, then pandas' C parser does the work -- ~20 s for the full
    match instead of minutes of Python string splitting.
    """
    import io
    with open(path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64, na_values=["NaN", "nan"])
    del text
    n_slots = (table.shape[1] - 1 - 3) // 3
    names = ["frame_id"] + ["s{}_{}".format(k, c) for k in range(1, n_slots + 1) for c in "xys"] \
        + ["ball_x", "ball_y", "ball_s"]
    table.columns = names
    frame_id = table.pop("frame_id").astype(np.int64)
    if not np.array_equal(frame_id.values, np.arange(1, len(table) + 1)):
        raise ValueError("ATD frame_ids are not 1..n contiguous -- see starter/load_atd.py")
    frame = pd.DataFrame({"frame": (frame_id - 1).astype(np.int64)})   # video frame
    return pd.concat([frame, table.astype(np.float32)], axis=1)


@dataclass
class Tracking:
    """Tracking arrays indexed by video frame. Positions normalised 0-1 (as in the file)."""
    frame: np.ndarray            # (n,) video frame
    xy: np.ndarray               # (n, slots, 2) normalised, NaN = not tracked
    s: np.ndarray                # (n, slots) raw 's' channel: player speed in m/s (the metadata's
                                 # 'kph' label is wrong -- verified, notebook 11 section 4)
    ball_xy: np.ndarray          # (n, 2) normalised, NaN = ball not tracked
    ball_s: np.ndarray           # (n,) raw ball 's' channel: NOT reliable (lower than the speed from
                                 # positions, partly missing) -- use load_signals()['ball_speed_mps']
    slot_team: np.ndarray        # (slots,) HOME / AWAY
    meta: dict = field(default_factory=dict)

    @property
    def xy_m(self):
        return self.xy * np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)

    @property
    def ball_m(self):
        return self.ball_xy * np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)

    @property
    def n_slots(self):
        return self.xy.shape[1]


def load_tracking(source="tactical", refresh=False):
    """The ATD of one source as a Tracking object. Parsed once, then read from parquet (~2 s)."""
    raw = games.require(["atd_raw"], data_dir=data_dir(), source=source)[0]
    table = _cached("atd", source, [raw], lambda: _parse_atd_raw(raw), refresh)
    meta = load_meta(source)
    n_slots = len(meta["slot_ids"])
    cols = ["s{}_{}".format(k, c) for k in range(1, n_slots + 1) for c in "xys"]
    block = table[cols].to_numpy(dtype=np.float32).reshape(len(table), n_slots, 3)
    return Tracking(frame=table["frame"].to_numpy(), xy=block[:, :, :2], s=block[:, :, 2],
                    ball_xy=table[["ball_x", "ball_y"]].to_numpy(dtype=np.float32),
                    ball_s=table["ball_s"].to_numpy(dtype=np.float32),
                    slot_team=meta["slot_team"], meta=meta)


# -- homography and Smart Tagging ------------------------------------------------------------------

def load_homography(source="tactical"):
    """(matrices (n,3,3), reprojection_error (n,), valid (n,) bool), row = video frame."""
    from load_homography import load
    return load(data_dir=data_dir(), source=source)


def load_smart_tagging():
    """One row per tagged interval, frames as video frames (end_frame exclusive).

    Adds `side` = H / A / None (Team N/A or unknown) and `half` = 1 / 2. Tag columns are the
    XML's own group names (Team, Half, Type, Side, ...).
    """
    from load_smart_tagging import load
    events = pd.DataFrame(load(data_dir=data_dir()))
    tags = events.pop("tags").apply(pd.Series)
    events = pd.concat([events, tags], axis=1)
    team = events.get("Team", pd.Series(index=events.index, dtype=object))
    events["side"] = team.fillna("").str.strip().str.lower().map(ST_TEAM_SIDE)
    unknown = sorted(set(team.dropna()) - {"N/A"}
                     - {t for t in team.dropna() if t.strip().lower() in ST_TEAM_SIDE})
    if unknown:
        print("warning: Team values not mapped to a side:", unknown)
    events["half"] = events.get("Half", pd.Series(index=events.index, dtype=object)).map(
        {"1st Half": 1, "2nd Half": 2})
    events["duration_s"] = events["end_s"] - events["start_s"]
    return events


# -- labels, segments, scoring ---------------------------------------------------------------------

def load_labels(path):
    """A frame,label CSV (ground truth or submission) as an array of 'H'/'A'/'D'/'X'."""
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, N_FRAMES)
    if reason:
        raise ValueError("{}: {}".format(path, reason))
    return np.array([str(v).strip().upper() for v in table["label"].values], dtype="<U1")


def load_gt_public():
    return load_labels(GT_PUBLIC_PATH)


def load_baselines():
    """{name: labels} for the four committed baselines."""
    names = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")
    return {name: load_labels(os.path.join(BASELINE_DIR, name + ".csv")) for name in names}


def codes(labels):
    return pm.label_codes(labels)


def graded_runs(gt_labels):
    """Half-open [start, stop) spans where the ground truth is not X (the scorer's own runs)."""
    return pm.valid_runs(codes(gt_labels))


def segments_table(labels, runs=None):
    """Constant-label segments as a DataFrame: start, stop (exclusive), label, frames, seconds.

    `runs` restricts to graded spans (default: the whole array, as one run).
    """
    runs = runs if runs is not None else [(0, len(labels))]
    rows = [(a, b, LABELS[k]) for a, b, k in pm.segments(codes(labels), runs)]
    table = pd.DataFrame(rows, columns=["start", "stop", "label"])
    table["frames"] = table["stop"] - table["start"]
    table["seconds"] = table["frames"] / FPS
    table["prev"] = table["label"].shift(1)
    table["next"] = table["label"].shift(-1)
    return table


def score(pred_labels, gt_labels=None):
    """pm.evaluate on label arrays (default ground truth: the public window)."""
    gt_labels = load_gt_public() if gt_labels is None else gt_labels
    return pm.evaluate(codes(gt_labels), codes(pred_labels))


def fill_abstentions(pred_labels, gt_labels):
    """The prediction as spellF1 sees it: X forward-filled inside each graded run."""
    filled = pm.fill_abstentions(codes(pred_labels), graded_runs(gt_labels))
    return np.array(LABELS, dtype="<U1")[filled]


# -- per-frame signals ------------------------------------------------------------------------------

def _speed_from_positions(xy_m):
    """m/s from consecutive positions (central where possible), NaN across tracking gaps."""
    velocity = np.full(xy_m.shape, np.nan, dtype=np.float32)
    velocity[1:-1] = (xy_m[2:] - xy_m[:-2]) * (FPS / 2.0)
    return np.linalg.norm(velocity, axis=-1)


def _signals(trk):
    n = len(trk.frame)
    xy, ball = trk.xy_m, trk.ball_m
    present = np.isfinite(xy[:, :, 0])
    ball_seen = np.isfinite(ball[:, 0])
    out = {"frame": trk.frame, "ball_visible": ball_seen,
           "n_tracked": present.sum(1).astype(np.int16)}

    # Nearest tracked player of each team, and overall.
    dist = np.sqrt(((xy - ball[:, None, :]) ** 2).sum(-1))           # (n, slots), NaN-safe below
    for team, name in ((HOME, "home"), (AWAY, "away")):
        d = np.where(present[:, trk.slot_team == team], dist[:, trk.slot_team == team], np.inf)
        best = d.min(1)
        out["nearest_{}_m".format(name)] = np.where(np.isfinite(best), best, np.nan)
        out["n_{}".format(name)] = present[:, trk.slot_team == team].sum(1).astype(np.int16)
    home, away = out["nearest_home_m"], out["nearest_away_m"]
    out["nearest_team"] = np.where(np.isnan(home) & np.isnan(away), NOBODY,
                                   np.where(np.nan_to_num(home, nan=np.inf)
                                            <= np.nan_to_num(away, nan=np.inf), HOME, AWAY))
    out["nearest_m"] = np.fmin(home, away)
    out["nearest_margin_m"] = away - home       # > 0: home closer; NaN if either team missing

    # Ball kinematics and position relative to the lines (metres; negative = outside).
    out["ball_speed_mps"] = _speed_from_positions(ball[:, None, :])[:, 0]
    out["ball_s_raw"] = trk.ball_s
    out["ball_x_m"], out["ball_y_m"] = ball[:, 0], ball[:, 1]
    out["ball_to_touchline_m"] = np.fmin(ball[:, 1], PITCH_WIDTH - ball[:, 1])
    out["ball_to_goalline_m"] = np.fmin(ball[:, 0], PITCH_LENGTH - ball[:, 0])
    out["ball_outside"] = ball_seen & ((ball[:, 0] < 0) | (ball[:, 0] > PITCH_LENGTH)
                                       | (ball[:, 1] < 0) | (ball[:, 1] > PITCH_WIDTH))

    # Team shape.
    for team, name in ((HOME, "home"), (AWAY, "away")):
        sel = trk.slot_team == team
        with np.errstate(invalid="ignore"), _quiet():
            cx, cy = np.nanmean(xy[:, sel, 0], 1), np.nanmean(xy[:, sel, 1], 1)
            spread = np.nanmean(np.sqrt((xy[:, sel, 0] - cx[:, None]) ** 2
                                        + (xy[:, sel, 1] - cy[:, None]) ** 2), 1)
        out["{}_cx_m".format(name)], out["{}_cy_m".format(name)] = cx, cy
        out["{}_spread_m".format(name)] = spread

    # Movement: from positions (m/s) and the raw 's' channel. Median over tracked slots, because
    # a slot that jumps to another person produces one absurd speed that would dominate a mean.
    with _quiet():
        out["players_speed_mps"] = np.nanmedian(_speed_from_positions(xy), 1)
        out["players_s_raw"] = np.nanmedian(np.where(present, trk.s, np.nan), 1)

    half = np.full(n, 0, dtype=np.int8)
    for number, first, last in HALVES:
        half[(trk.frame >= first) & (trk.frame <= last)] = number
    out["half"] = half
    return pd.DataFrame(out)


class _quiet:
    """Silence 'Mean of empty slice' -- an all-NaN frame legitimately has no mean."""
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def load_signals(source="tactical", refresh=False):
    """Per-frame signal table (one row per video frame), cached. See _signals for the columns."""
    raw = games.require(["atd_raw", "atd_meta"], data_dir=data_dir(), source=source)
    return _cached("signals", source, raw, lambda: _signals(load_tracking(source)), refresh)


# -- slot (tracklet) structure ---------------------------------------------------------------------

def slot_pieces(trk):
    """Every contiguous stretch a slot is tracked: slot, team, start, stop (excl.), frames."""
    rows = []
    present = np.isfinite(trk.xy[:, :, 0])
    for slot in range(trk.n_slots):
        edges = np.diff(np.concatenate(([0], present[:, slot].astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            rows.append((slot + 1, int(trk.slot_team[slot]), int(trk.frame[a]),
                         int(trk.frame[b - 1]) + 1))
    table = pd.DataFrame(rows, columns=["slot", "team", "start", "stop"])
    table["frames"] = table["stop"] - table["start"]
    table["seconds"] = table["frames"] / FPS
    return table


def runs_of(mask):
    """[(start, stop)] of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


# -- plotting --------------------------------------------------------------------------------------

def plot_label_strip(ax, labels, start, stop, y=0.0, height=0.8, name=None):
    """Draw labels[start:stop] as coloured bars on ax, x axis in video frames."""
    window = np.asarray(labels[start:stop])
    for a, b, k in pm.segments(codes(window), [(0, len(window))]):
        ax.broken_barh([(start + a, b - a)], (y, height), color=LABEL_COLOURS[LABELS[k]],
                       edgecolor="none")
    if name is not None:
        ax.text(start, y + height / 2, " " + name, va="center", ha="right", fontsize=9)


def plot_timeline(rows, start, stop, ax=None, title=None):
    """Stacked label strips, top to bottom, e.g. {'GT': gt, 'smooth50': pred}."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 0.55 * len(rows) + 1.0))
    for i, (name, labels) in enumerate(rows.items()):
        plot_label_strip(ax, labels, start, stop, y=len(rows) - 1 - i, name=name)
    ax.set_xlim(start, stop)
    ax.set_ylim(-0.2, len(rows))
    ax.set_yticks([])
    ax.set_xlabel("video frame")
    secax = ax.secondary_xaxis("top", functions=(lambda f: (f - start) / FPS,
                                                 lambda s: s * FPS + start))
    secax.set_xlabel("seconds from frame {}".format(start))
    ax.legend(handles=[Patch(color=LABEL_COLOURS[k], label=k) for k in "HAD"],
              loc="upper right", bbox_to_anchor=(1.0, -0.35), ncol=3, frameon=False)
    if title:
        ax.set_title(title)
    return ax


def shade_labels(ax, labels, start, stop, alpha=0.15):
    """Background colour by label on an ax whose x axis is video frames."""
    window = np.asarray(labels[start:stop])
    for a, b, k in pm.segments(codes(window), [(0, len(window))]):
        if LABELS[k] != "X":
            ax.axvspan(start + a, start + b, color=LABEL_COLOURS[LABELS[k]], alpha=alpha, lw=0)
