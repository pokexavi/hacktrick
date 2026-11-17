"""Helpers of the Dead-ball review page: what each model folder is, the match's known events
(Smart Tagging) as frame windows, and what every model predicts around them.

No Streamlit here, so it can be tested on hand-built tables (test_dead_ball_review.py).
"""
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

# Graded halves of BAR-ATH: first frame of each half (the kickoffs) in the tactical video.
KICKOFF = {1: 11185, 2: 113742}
# The halves as played, [first, last + 1): from each kickoff to the last graded frame. The graded
# frames of the second half before its kickoff (110808-113741) are all D by the rules, so the
# review leaves them out, with halftime and everything outside the match.
PLAYED = [(11185, 90652), (113742, 190198)]
SMART_TAGGING_KINDS = {"Corner Kick": "corner", "Free Kick": "free_kick", "Goal Kick": "goal_kick",
                       "Throw-in": "throw_in", "Kick Off (After Goal)": "kickoff_after_goal",
                       "Kick Off (Start)": "kickoff"}


def in_play(n):
    """(n,) bool: the frames of the halves as played."""
    mask = np.zeros(n, dtype=bool)
    for a, b in PLAYED:
        mask[a:min(b, n)] = True
    return mask


def half_bounds(frame):
    """[first, last + 1) of the played half that holds (or is nearest to) a frame."""
    return min(PLAYED, key=lambda half: 0 if half[0] <= frame < half[1]
               else min(abs(frame - half[0]), abs(frame - half[1])))


def clip_to_play(table, fps=25):
    """Rows (start, stop) cut to the played halves; rows outside them are dropped. A 'seconds'
    column is recomputed."""
    if table is None or table.empty:
        return table
    parts = []
    for a, b in PLAYED:
        part = table[(table["stop"] > a) & (table["start"] < b)].copy()
        part["start"] = part["start"].clip(lower=a)
        part["stop"] = part["stop"].clip(upper=b)
        parts.append(part)
    out = pd.concat(parts).sort_values("start")
    if "seconds" in out:
        out["seconds"] = (out["stop"] - out["start"]) / fps
    return out.reset_index(drop=True)


def describe_model(folder):
    """One row about a model folder: its heads, the type output and the decoder (config.toml,
    model.json). Missing files give empty cells (older folders)."""
    folder = Path(folder)
    cfg, info = {}, {}
    if (folder / "config.toml").exists():
        with open(folder / "config.toml", "rb") as handle:
            cfg = tomllib.load(handle)
    if (folder / "model.json").exists():
        info = json.loads((folder / "model.json").read_text())
    types = cfg.get("types", {})
    classes = types.get("classes", [])
    typed = cfg.get("decoding", {}).get("typed", {}).get("enabled", False)
    if not cfg:
        heads = "?"
    elif classes:
        heads = "dead/start/stop + type ({} classes, weight {:g})".format(len(classes), types.get("weight", 0))
    else:
        heads = "dead/start/stop"
    decoder = "typed HSMM" if typed else "HSMM"
    if "temperature" in info:
        decoder += " (T {:g}, w_start {:g}, w_stop {:g}".format(
            info["temperature"], info.get("w_start", 0), info.get("w_stop", 0))
        decoder += ", type weight {:g})".format(info["type_weight"]) if typed else ")"
    return {"model": folder.name, "trained on": cfg.get("augment", {}).get("train", {}).get("recipe", "?"),
            "outputs": heads, "classes": ", ".join(classes), "decoder": decoder,
            "heads file": info.get("heads") or info.get("oof") or info.get("run", "")}


def smart_tagging_events(events, fps=25):
    """Set pieces and goals of the Smart Tagging (parse_events output) as frame windows. Its
    set pieces are fixed ~35 s windows around the restart; repeated tags are dropped."""
    rows = []
    for event in events:
        kind = None
        if event["code"] == "SET PIECES":
            kind = SMART_TAGGING_KINDS.get(event["tags"].get("Type"))
        elif event["code"] == "GOALS":
            kind = "goal"
        if kind is None:
            continue
        half = event["tags"].get("Half", "")
        minute = (event["start"] - KICKOFF[2 if half.startswith("2") else 1]) / fps / 60
        rows.append({"kind": kind, "team": event["tags"].get("Team", ""),
                     "minute": "{:.0f}'".format(minute + (45 if half.startswith("2") else 0) + 1),
                     "description": "{} ({})".format(event["tags"].get("Type", event["code"]),
                                                     event["tags"].get("Team", "")),
                     "start": int(event["start"]), "stop": int(event["end"])})
    table = pd.DataFrame(rows, columns=["kind", "team", "minute", "description", "start", "stop"])
    return table.drop_duplicates(["kind", "start", "stop"]).sort_values("start").reset_index(drop=True)


def overlapping(stops, start, stop):
    """The model's predicted stoppages that overlap [start, stop)."""
    return stops[(stops["stop"] > start) & (stops["start"] < stop)]


def without_stoppage(events, stops):
    """The events whose window [start, stop) no predicted stoppage overlaps, not even by a frame."""
    empty = [overlapping(stops, a, b).empty for a, b in zip(events["start"], events["stop"])]
    return events[np.array(empty, dtype=bool)].copy()


def summarise(stops, start, stop):
    """What a model predicts in [start, stop), in a few words: '1 · 122 s goal',
    '3 · 8+18+70 s', or 'none'. Whole stoppages, even when they stick out of the window."""
    part = overlapping(stops, start, stop)
    if part.empty:
        return "none"
    seconds = "+".join("{:.0f}".format(s) for s in part["seconds"])
    text = "{} · {} s".format(len(part), seconds)
    if "type" in part:
        kinds = part["type"].fillna("").astype(str)
        kinds = [k for k in dict.fromkeys(kinds) if k]
        if kinds:
            text += " " + "/".join(kinds)
    return text


def coverage(events, models):
    """Per event kind and model: share of events with a predicted stoppage overlapping the
    window, and the mean number of stoppages there (> 1 = the stoppage is cut in pieces)."""
    rows = []
    for kind, part in events.groupby("kind"):
        row = {"kind": kind, "events": len(part)}
        for name, stops in models.items():
            counts = [len(overlapping(stops, a, b)) for a, b in zip(part["start"], part["stop"])]
            row[name] = "{:.0%} · {:.1f}".format(np.mean([c > 0 for c in counts]), np.mean(counts))
        rows.append(row)
    return pd.DataFrame(rows)


def disagreement_kind(dead, start, stop):
    """Kind of a run [start, stop) where the models (dead: (models, frames) bool) disagree, from
    what all of them say just around it: inside a stoppage they all see, at the edge of one
    (a boundary placed differently), or a stoppage only some models see."""
    n = dead.shape[1]
    before = start > 0 and dead[:, start - 1].all()
    after = stop < n and dead[:, stop].all()
    if before and after:
        return "gap inside a stoppage"
    if before or after:
        return "boundary shifted"
    return "stoppage seen by some models"


def types_in(stops, start, stop):
    """Predicted types of the stoppages overlapping [start, stop) (empty without a type output)."""
    if "type" not in stops:
        return []
    return [str(k) for k in overlapping(stops, start, stop)["type"].dropna().unique()]
