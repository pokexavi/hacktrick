"""Readers for the Metrica open data, adapted to the viewer's formats.

It has no ball-status label like DFL's, so each tracking dict carries an estimate: dead where the
ball is missing for at least MIN_DEAD_S. Shorter gaps are tracker dropouts, so those frames stay
alive.

Events are translated into the tags stoppages.py links to dead-ball intervals: set_piece_type,
FOUL_COMMITTED, BALL_OUT, GENERIC:Offside and SHOT with result GOAL.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from config import METRICA_DIR

METRICA_FPS = 25
MIN_DEAD_S = 1.0
METRICA_LABEL_NOTE = (f"Dead ball is an estimate: the ball is missing for at least {MIN_DEAD_S:.0f} s. "
                      "It comes from the tracking itself, so it is not independent of ball visibility.")
METRICA_RESTARTS = {"FREE KICK": "FREE_KICK", "THROW IN": "THROW_IN", "CORNER KICK": "CORNER_KICK",
                    "KICK OFF": "KICK_OFF", "GOAL KICK": "GOAL_KICK", "PENALTY": "PENALTY"}
POSSESSION_UNKNOWN, POSSESSION_HOME, POSSESSION_AWAY, POSSESSION_DEAD = -1, 0, 1, 2
# Metrica events that say which team has the ball. Challenges involve both teams, and the rest
# (fouls, cards, ball out) say nothing about who holds it.
METRICA_ON_BALL_EVENTS = ("PASS", "CARRY", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE")
EVENT_COLUMNS = ["event_id", "event_type", "period_id", "timestamp_s", "end_timestamp_s", "team_id",
                 "player_id", "set_piece_type", "result"]


def open_sources():
    """Every Metrica match whose files are all on disk."""
    available = {}
    for game in (1, 2):
        folder = METRICA_DIR / "data" / f"Sample_Game_{game}"
        stem = f"Sample_Game_{game}_"
        source = {"kind": "metrica_csv", "dataset": "Metrica", "match": f"Sample_Game_{game}",
                  "tracking": folder / f"{stem}RawTrackingData_Home_Team.csv",
                  "away": folder / f"{stem}RawTrackingData_Away_Team.csv",
                  "meta": None, "events": folder / f"{stem}RawEventsData.csv"}
        if source["tracking"].is_file() and source["away"].is_file():
            available[f"Metrica · Sample game {game}"] = source
    folder = METRICA_DIR / "data" / "Sample_Game_3"
    source = {"kind": "metrica_epts", "dataset": "Metrica", "match": "Sample_Game_3",
              "tracking": folder / "Sample_Game_3_tracking.txt",
              "meta": folder / "Sample_Game_3_metadata.xml",
              "events": folder / "Sample_Game_3_events.json"}
    if source["tracking"].is_file() and source["meta"].is_file():
        available["Metrica · Sample game 3"] = source
    return available


def _period_times(frame, period, fps):
    """Seconds from the first frame of each period, the convention of the DFL matches."""
    times = np.full(len(frame), np.nan, dtype=np.float32)
    for value in np.unique(period):
        rows = np.flatnonzero(period == value)
        times[rows] = (frame[rows] - frame[rows[0]]) / fps
    return times


def _ball_missing_labels(missing, period, fps):
    """1 on missing-ball runs of at least MIN_DEAD_S within a period, else 0."""
    labels = np.zeros(len(missing), dtype=np.int8)
    breaks = np.flatnonzero(period[1:] != period[:-1]) + 1
    for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(missing)]):
        edges = np.flatnonzero(np.diff(np.r_[0, missing[start:stop].astype(np.int8), 0]))
        for run_start, run_stop in zip(edges[0::2], edges[1::2]):
            if (run_stop - run_start) / fps >= MIN_DEAD_S:
                labels[start + run_start:start + run_stop] = 1
    return labels


def _frame_seconds(frames, periods, starts, fps):
    return (frames.to_numpy(np.float64) - periods.map(starts).to_numpy(np.float64)) / fps


def _finish_events(events):
    """The viewer's column types, in time order. Empty set-piece and result cells become NA."""
    events = events[EVENT_COLUMNS].copy()
    events["period_id"] = events.period_id.astype(int)
    for column in ("timestamp_s", "end_timestamp_s"):
        events[column] = events[column].astype(float)
    for column in ("event_type", "team_id", "player_id", "set_piece_type", "result"):
        events[column] = events[column].astype("string")
    return events.sort_values(["period_id", "timestamp_s"], kind="stable").reset_index(drop=True)


# -- Metrica ---------------------------------------------------------------------------------------

def _metrica_side(path):
    table = pd.read_csv(path, skiprows=2)
    players = table.iloc[:, 3:-2].to_numpy(np.float32)
    return table, players.reshape(len(table), -1, 2)


@st.cache_resource(show_spinner="Loading Metrica tracking...")
def load_metrica_csv_tracking(home_path, away_path, modified, away_modified):
    home, home_xy = _metrica_side(home_path)
    away, away_xy = _metrica_side(away_path)
    if not home.Frame.equals(away.Frame):
        raise ValueError("Home and away tracking files do not share their frames")
    return _metrica_tracking(home.Frame.to_numpy(np.int64), home.Period.to_numpy(np.int8),
                             home.iloc[:, -2:].to_numpy(np.float32),
                             np.concatenate((home_xy, away_xy), axis=1),
                             home_xy.shape[1], away_xy.shape[1], ("Home", "Away"), "Metrica CSV")


@st.cache_resource(show_spinner="Loading Metrica tracking...")
def load_metrica_epts_tracking(meta_path, raw_path, meta_modified, raw_modified):
    from kloppy import metrica
    dataset = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path, coordinates="kloppy")
    table = dataset.to_df()
    teams = list(dataset.metadata.teams)
    slots = [str(player.player_id) for team in teams for player in team.players]
    xy = np.full((len(table), len(slots), 2), np.nan, dtype=np.float32)
    for index, slot in enumerate(slots):
        for axis, column in enumerate((f"{slot}_x", f"{slot}_y")):
            if column in table:
                xy[:, index, axis] = pd.to_numeric(table[column], errors="coerce").to_numpy(np.float32)
    ball = table[["ball_x", "ball_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return _metrica_tracking(pd.to_numeric(table.frame_id).to_numpy(np.int64),
                             table.period_id.to_numpy(np.int8), ball, xy,
                             len(teams[0].players), len(teams[1].players),
                             (teams[0].name, teams[1].name), "Metrica EPTS")


def _metrica_tracking(frame, period, ball, players, n_home, n_away, team_names, source_format):
    """Metrica's 0-1 coordinates, origin top left, are already the viewer's."""
    return {"frame_id": frame, "period_id": period,
            "timestamp_s": _period_times(frame, period, METRICA_FPS),
            "ball_xy": ball, "ball_z": np.full(len(frame), np.nan, dtype=np.float32),
            "players_xy": players,
            "player_team": np.r_[np.zeros(n_home, np.int8), np.ones(n_away, np.int8)],
            "team_names": np.array(team_names),
            "label": _ball_missing_labels(np.isnan(ball[:, 0]), period, METRICA_FPS),
            "fps": METRICA_FPS, "source_format": source_format, "label_note": METRICA_LABEL_NOTE}


def metrica_possession(tracking, events):
    """Per-frame possession estimate: the team of the last on-ball event, grey over dead ball."""
    periods, times = tracking["period_id"], tracking["timestamp_s"]
    teams = {str(name): code for code, name in enumerate(tracking["team_names"])}
    possession = np.full(len(periods), POSSESSION_UNKNOWN, dtype=np.int8)
    on_ball = events[events.event_type.isin(METRICA_ON_BALL_EVENTS)]
    for period, group in on_ball.groupby("period_id"):
        rows = np.flatnonzero(periods == period)
        steps = np.rint(group.timestamp_s.to_numpy() * tracking["fps"]).astype(np.int64)
        starts = np.clip(steps, 0, len(rows) - 1)
        codes = group.team_id.map(teams).to_numpy()
        for start, stop, code in zip(starts, np.r_[starts[1:], len(rows)], codes):
            possession[rows[start:stop]] = code
    possession[tracking["label"] == 1] = POSSESSION_DEAD
    return possession


@st.cache_data(show_spinner="Loading Metrica events...")
def load_metrica_csv_events(path, period_starts, modified):
    table = pd.read_csv(path)
    return _metrica_events(pd.DataFrame({
        "type": table["Type"], "subtype": table["Subtype"].fillna(""), "period": table["Period"],
        "frame": table["Start Frame"], "end_frame": table["End Frame"], "team": table["Team"],
        "player": table["From"]}), period_starts)


@st.cache_data(show_spinner="Loading Metrica events...")
def load_metrica_json_events(path, period_starts, modified):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)["data"]
    return _metrica_events(pd.DataFrame([{
        "type": row["type"]["name"], "subtype": _json_subtype(row.get("subtypes")),
        "period": row["period"], "frame": row["start"]["frame"], "end_frame": row["end"]["frame"],
        "team": row["team"]["name"], "player": (row.get("from") or {}).get("name")}
        for row in rows]), period_starts)


def _json_subtype(value):
    """One subtype or a list of them, joined as the CSV games write them: 'ON TARGET-GOAL'."""
    values = value if isinstance(value, list) else [value] if value else []
    return "-".join(item["name"] for item in values)


def _metrica_restart(row):
    """A goal kick is tagged on the pass that takes it. The other restarts are SET PIECE rows."""
    if "GOAL KICK" in row.subtype:
        return "GOAL_KICK"
    if row.type == "SET PIECE":
        return next((tag for name, tag in METRICA_RESTARTS.items() if name in row.subtype), None)
    return None


def _metrica_events(table, period_starts):
    table = table[table.period.isin(period_starts)].reset_index(drop=True)
    end_frame = table.end_frame.where(table.end_frame >= table.frame, table.frame)
    is_goal = table.type.eq("SHOT") & table.subtype.str.contains("GOAL")
    is_offside = table.subtype.str.contains("OFFSIDE")
    event_type = table.type.replace({"BALL OUT": "BALL_OUT", "FAULT RECEIVED": "FOUL_COMMITTED"})
    return _finish_events(pd.DataFrame({
        "event_id": table.index.astype(str),
        "event_type": event_type.where(~is_offside, "GENERIC:Offside"),
        "period_id": table.period,
        "timestamp_s": _frame_seconds(table.frame, table.period, period_starts, METRICA_FPS),
        "end_timestamp_s": _frame_seconds(end_frame, table.period, period_starts, METRICA_FPS),
        "team_id": table.team, "player_id": table.player,
        "set_piece_type": [_metrica_restart(row) for row in table.itertuples()],
        "result": table.subtype.where(~is_goal, "GOAL")}))
