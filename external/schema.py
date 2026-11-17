"""The one per-match format that every source is converted into, and the loader that reads it back.

Two converters write it -- fetch_dfl.py for the Sportec Open DFL matches and convert_atd.py for our
BAR-ATH tracking -- so that a model trained on one reads the other without a line of glue code.

The format
----------
One parquet file per match plus a JSON sidecar with the same stem. One row per frame at 25 fps:

    frame        int64    the source's own frame index (BAR-ATH: video frame = submission row)
    period       int8     1, 2 -- or 0 outside the halves (BAR-ATH pre-match, halftime, post-match)
    ball_x/y/z   float32  metres; NaN where the ball is not tracked. z is NaN for every BAR-ATH frame
    h{k}_x/y     float32  home slot k = 1..n_home, metres, NaN where the slot is empty
    a{k}_x/y     float32  away slot k = 1..n_away
    label        str      H / A / D, or X where the source has no label for that frame

Coordinates, identical for every source:

    origin       the centre spot
    x            along the pitch, metres. Rotated per period so that HOME ALWAYS ATTACKS +x
    y            across the pitch, metres, +y on the left of a home attacker
    pitch        the true size of that pitch, in the sidecar (DFL pitches are not all 105 x 68)

A slot is a column pair, not a person. In DFL it is one player for the whole match; in BAR-ATH it
is one tracker slot that drops, reappears and changes person. A model should therefore treat the
players of a team as an unordered set -- which also makes the two sources look alike.

Velocities are deliberately not stored: the ATD's own speed channel is unreliable and DFL's is in
other units, so both are recomputed from positions at training time, the same way.

Usage:
    from external.schema import load_match, processed_paths
    for path in processed_paths("dfl"):
        match = load_match(path)
        match.home.shape            # (frames, n_home, 2)
"""

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Params
SCHEMA_VERSION = 1
FPS = 25
LABELS = ("H", "A", "D", "X")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Where each source's processed files live. DFL is public (CC-BY 4.0) and sits under external/;
# BAR-ATH is competition data and stays inside the gated data directory, never anywhere else.
DFL_DIR = os.environ.get("DFL_DATA_DIR") or os.path.join(HERE, "dfl")
# Seconds after a kickoff over which the team centroids decide the direction of play. Both teams
# are in their own half at a kickoff, so a couple of seconds is unambiguous.
ORIENTATION_WINDOW_S = 2


def bar_processed_dir():
    """data/processed, or $MTAH_DATA_DIR/processed. Read at call time, like exploration/common."""
    return os.path.join(os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data"), "processed")


def processed_paths(source="dfl"):
    """Sorted parquet paths of every processed match of a source ('dfl' or 'bar')."""
    folder = os.path.join(DFL_DIR, "processed") if source == "dfl" else bar_processed_dir()
    return sorted(glob.glob(os.path.join(folder, "*.parquet")))


def player_columns(side, n_slots):
    """['h1_x', 'h1_y', 'h2_x', ...] for side 'h' or 'a'."""
    return ["{}{}_{}".format(side, k, c) for k in range(1, n_slots + 1) for c in "xy"]


# -- orientation -----------------------------------------------------------------------------------

def home_attacks_positive(home_xy, away_xy):
    """True if, over these kickoff frames, the home team stands on the -x half (so attacks +x).

    home_xy, away_xy: (frames, slots, 2) in a centre-origin frame. None if either team is absent.
    """
    with np.errstate(invalid="ignore"):
        home = np.nanmean(home_xy[..., 0]) if np.isfinite(home_xy[..., 0]).any() else np.nan
        away = np.nanmean(away_xy[..., 0]) if np.isfinite(away_xy[..., 0]).any() else np.nan
    if np.isnan(home) or np.isnan(away):
        return None
    return bool(home < away)


def rotate_180(table, n_home, n_away, rows):
    """Rotate the given rows of a canonical table by 180 degrees in place (x -> -x, y -> -y).

    A rotation, not a mirror: it keeps left and right where a player sees them.
    """
    cols = ["ball_x", "ball_y"] + player_columns("h", n_home) + player_columns("a", n_away)
    table.loc[rows, cols] = -table.loc[rows, cols]


# -- writing and reading ---------------------------------------------------------------------------

def write_match(table, meta, out_dir, name):
    """Validate, then write <out_dir>/<name>.parquet and <name>.json. Returns the parquet path."""
    problems = validate(table, meta)
    if problems:
        raise ValueError("{}: not written --\n  ".format(name) + "\n  ".join(problems))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".parquet")
    table.to_parquet(path, index=False)
    with open(os.path.join(out_dir, name + ".json"), "w") as handle:
        json.dump(dict(meta, schema_version=SCHEMA_VERSION), handle, indent=2, ensure_ascii=False)
    return path


def validate(table, meta):
    """Structural problems with a canonical table, as a list of strings. Empty means valid.

    Checks shape and conventions only -- never values that would differ by source.
    """
    problems = []
    n_home, n_away = meta["n_home"], meta["n_away"]
    expected = (["frame", "period", "ball_x", "ball_y", "ball_z"]
                + player_columns("h", n_home) + player_columns("a", n_away) + ["label"])
    if list(table.columns) != expected:
        missing = sorted(set(expected) - set(table.columns))
        extra = sorted(set(table.columns) - set(expected))
        problems.append("columns differ from the schema (missing {}, extra {}, or order)".format(
            missing[:4], extra[:4]))
        return problems
    if not table["frame"].is_monotonic_increasing or table["frame"].duplicated().any():
        problems.append("frame is not strictly increasing")
    if not set(table["period"].unique()) <= {0, 1, 2}:
        problems.append("period has values outside 0/1/2")
    if not set(table["label"].unique()) <= set(LABELS):
        problems.append("label has values outside H/A/D/X")
    if (table.loc[table["period"] == 0, "label"] != "X").any():
        problems.append("frames outside the halves carry a label")

    # Everything tracked must be on or near the pitch it belongs to. A wrong unit (0-1 instead of
    # metres) or a missing centring shows up here immediately.
    length, width = meta["pitch"]
    xs = table[["ball_x"] + player_columns("h", n_home)[0::2] + player_columns("a", n_away)[0::2]]
    ys = table[["ball_y"] + player_columns("h", n_home)[1::2] + player_columns("a", n_away)[1::2]]
    margin = 15.0       # balls and players do leave the pitch -- behind goals, at throw-ins
    if np.nanmax(np.abs(xs.to_numpy())) > length / 2 + margin:
        problems.append("x reaches beyond the pitch by more than {} m".format(margin))
    if np.nanmax(np.abs(ys.to_numpy())) > width / 2 + margin:
        problems.append("y reaches beyond the pitch by more than {} m".format(margin))
    return problems


@dataclass
class Match:
    """One processed match as arrays. All arrays share the first axis (frames)."""
    name: str
    source: str
    frame: np.ndarray            # (T,) int64
    period: np.ndarray           # (T,) int8
    ball: np.ndarray             # (T, 3) float32, metres, NaN = not tracked
    home: np.ndarray             # (T, n_home, 2) float32, metres, NaN = slot empty
    away: np.ndarray             # (T, n_away, 2) float32
    label: np.ndarray            # (T,) '<U1' in H/A/D/X
    meta: dict = field(default_factory=dict)

    @property
    def pitch(self):
        return tuple(self.meta["pitch"])

    def label_codes(self):
        """Labels as 0..3 in LABELS order -- the same codes evaluation/possession_metrics uses."""
        codes = np.full(len(self.label), LABELS.index("X"), dtype=np.int8)
        for code, value in enumerate(LABELS):
            codes[self.label == value] = code
        return codes


def load_match(path):
    """Read a processed parquet (and its sidecar) back into a Match."""
    stem = os.path.splitext(path)[0]
    with open(stem + ".json") as handle:
        meta = json.load(handle)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("{}: schema version {} != {} -- re-run its converter".format(
            path, meta.get("schema_version"), SCHEMA_VERSION))
    table = pd.read_parquet(path)
    n_home, n_away, n = meta["n_home"], meta["n_away"], len(table)
    return Match(
        name=os.path.basename(stem), source=meta["source"],
        frame=table["frame"].to_numpy(np.int64), period=table["period"].to_numpy(np.int8),
        ball=table[["ball_x", "ball_y", "ball_z"]].to_numpy(np.float32),
        home=table[player_columns("h", n_home)].to_numpy(np.float32).reshape(n, n_home, 2),
        away=table[player_columns("a", n_away)].to_numpy(np.float32).reshape(n, n_away, 2),
        label=table["label"].to_numpy().astype("<U1"), meta=meta)


# -- possession spells -----------------------------------------------------------------------------

def spells(label, period):
    """[(start, stop, label)] maximal runs of one label, never crossing a period boundary.

    Same definition as the scorer: H H D D H H is two possessions.
    """
    label = np.asarray(label)
    period = np.asarray(period)
    change = np.flatnonzero((label[1:] != label[:-1]) | (period[1:] != period[:-1])) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(label)]))
    return [(int(a), int(b), str(label[a])) for a, b in zip(starts, stops)]


def possession_lengths(label, period):
    """Lengths in frames of every H or A spell."""
    return np.array([b - a for a, b, k in spells(label, period) if k in ("H", "A")], dtype=np.int64)
