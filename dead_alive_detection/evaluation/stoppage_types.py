"""
Labels each true DFL stoppage with its kind, from the Sportec events, and measures recall per kind.

The coarse groups are whistle, ball out, corner, goal and other. The finer classes (FINE_CLASSES)
are used for the TCN's stoppage-type output.
"""

import os

import numpy as np
import pandas as pd

from dead_alive_detection.evaluation.metrics import dead_segments, pm, scored_runs
from external import schema

GROUPS = ("whistle", "ball out", "corner", "goal", "other")
RESTARTS = {"CORNER_KICK": "corner", "GOAL_KICK": "ball out", "THROW_IN": "ball out",
            "FREE_KICK": "whistle", "PENALTY": "whistle", "KICK_OFF": "goal"}
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
FINE_CLASSES = ("whistle", "throw_in", "goal_kick", "corner", "goal", "incident", "other")
INCIDENT_EVENTS = ("CARD", "SUBSTITUTION", "GENERIC:VideoAssistantAction", "GENERIC:RefereeBall",
                   "GENERIC:CautionTeamofficial")


def event_paths(name):
    """
    Match-information and raw-events XML paths of a DFL match, or None if not downloaded.
    A match from another source has a single events CSV instead.
    """
    raw = os.path.join(schema.DFL_DIR, "raw")
    table = os.path.join(raw, name + "_events.csv")
    if os.path.isfile(table):
        return (table,)
    found = {kind: [f for f in os.listdir(raw) if f.startswith(kind) and f.endswith(name + ".xml")]
             for kind in ("DFL_02_01_matchinformation", "DFL_03_02_events_raw")} \
        if os.path.isdir(raw) else {}
    if not found or not all(found.values()):
        return None
    return tuple(os.path.join(raw, found[kind][0])
                 for kind in ("DFL_02_01_matchinformation", "DFL_03_02_events_raw"))


def load_events(meta_path, events_path=None):
    """
    Loads the Sportec event feed as a table of event type, period, time, set piece and result.
    With a single path, reads that CSV instead.
    """
    if events_path is None:
        return pd.read_csv(meta_path)
    from kloppy import sportec
    events = sportec.load_event(meta_data=meta_path, event_data=events_path).to_df()
    events["timestamp_s"] = pd.to_timedelta(events["timestamp"]).dt.total_seconds()
    for column in ("set_piece_type", "result"):
        if column not in events:
            events[column] = None
    return events[["event_type", "period_id", "timestamp_s", "set_piece_type", "result"]]


def classify(start_s, end_s, period, events):
    """
    The group of one stoppage, from the events at its start and at its restart.
    """
    nearby = events[(events.period_id == period) & events.timestamp_s.between(start_s - 5, end_s + 4)]
    onset = nearby[nearby.timestamp_s.between(start_s - 5, min(end_s, start_s + 10))]
    if (onset.event_type.eq("SHOT") & onset.result.astype(str).eq("GOAL")).any():
        return "goal"
    if onset.event_type.isin(["GENERIC:Offside", "FOUL_COMMITTED"]).any():
        return "whistle"
    restart = nearby[nearby.timestamp_s.between(end_s - 3, end_s + 4) & nearby.set_piece_type.notna()]
    if len(restart):
        nearest = restart.iloc[(restart.timestamp_s - end_s).abs().argmin()]
        return RESTARTS.get(str(nearest.set_piece_type), "other")
    if onset.event_type.eq("BALL_OUT").any():
        return "ball out"
    return "other"


def true_stoppages(match, truth):
    """
    The true dead segments of a DFL match with their group, or [] if its events are missing.
    """
    paths = event_paths(match.name)
    if paths is None:
        return []
    events = load_events(*paths)
    first = {p: np.flatnonzero(match.period == p)[0] for p in (1, 2)}
    out = []
    for a, b, _ in dead_segments(truth == pm.D, scored_runs(truth, match.period)):
        period = int(match.period[a])
        start_s, end_s = (a - first[period]) / schema.FPS, (b - first[period]) / schema.FPS
        out.append((a, b, classify(start_s, end_s, period, events)))
    return out


def recall_by_group(stoppages, pred_dead, truth, period):
    """
    Share of true stoppages found in each group, averaged over the IoU thresholds in IOUS.
    """
    pred = dead_segments(pred_dead, scored_runs(truth, period))
    found = {group: [] for group in GROUPS}
    for a, b, group in stoppages:
        best = max([(min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) for c, d, _ in pred
                    if c < b and d > a] or [0.0])
        found[group].append(np.mean([best >= iou for iou in IOUS]))
    return {group: float(np.mean(v)) if v else np.nan for group, v in found.items()}


def fine_class(start_s, end_s, period, events, classes=FINE_CLASSES):
    """
    The fine class of one stoppage, restricted to the given classes.
    """
    group = classify(start_s, end_s, period, events)
    nearby = events[(events.period_id == period) & events.timestamp_s.between(start_s - 5, end_s + 4)]
    restart = nearby[nearby.timestamp_s.between(end_s - 3, end_s + 4) & nearby.set_piece_type.notna()]
    restart = (str(restart.set_piece_type.iloc[(restart.timestamp_s - end_s).abs().argmin()])
               if len(restart) else "")
    if group == "goal":
        return "goal"
    if "incident" in classes:
        inside = events[(events.period_id == period) & events.timestamp_s.between(start_s, end_s)]
        if restart == "PENALTY" or inside.event_type.isin(INCIDENT_EVENTS).any():
            return "incident"
    if group == "ball out":
        fine = {"THROW_IN": "throw_in", "GOAL_KICK": "goal_kick"}.get(restart, "other")
    else:
        fine = group
    return fine if fine in classes else "other"


def fine_stoppages(match, truth, classes=FINE_CLASSES):
    """
    The true dead segments of a DFL match with their fine class, or None if its events are missing.
    """
    paths = event_paths(match.name)
    if paths is None:
        return None
    events = load_events(*paths)
    first = {p: np.flatnonzero(match.period == p)[0] for p in (1, 2)}
    out = []
    for a, b, _ in dead_segments(truth == pm.D, scored_runs(truth, match.period)):
        period = int(match.period[a])
        start_s, end_s = (a - first[period]) / schema.FPS, (b - first[period]) / schema.FPS
        out.append((a, b, fine_class(start_s, end_s, period, events, classes)))
    return out


def step_classes(stoppages, rows, classes):
    """
    Class index per 5 Hz step, -1 outside stoppages.
    """
    frame_class = np.full(int(rows[-1]) + 1, -1, dtype=np.int8)
    for a, b, kind in stoppages:
        frame_class[a:b] = classes.index(kind)
    return frame_class[rows]
