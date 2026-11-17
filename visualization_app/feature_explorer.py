"""Read a small, aligned subset of the four feature CSVs for inspection."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
METRICS = {
    "Players": {
        "all_median_reported_speed_m_s": "All median speed (m/s)",
        "home_median_reported_speed_m_s": "Home median speed (m/s)",
        "away_median_reported_speed_m_s": "Away median speed (m/s)",
        "all_slow_fraction": "Slow player fraction",
        "all_stationary_fraction": "Stationary player fraction",
        "home_slots_observed": "Home slots observed",
        "away_slots_observed": "Away slots observed",
    },
    "Teams": {
        "centroid_distance_m": "Team centroid distance (m)",
        "nearest_opponent_distance_m": "Nearest opponent distance (m)",
        "home_team_spread_m": "Home spread (m)",
        "away_team_spread_m": "Away spread (m)",
        "home_hull_area_m2": "Home hull area (m²)",
        "away_hull_area_m2": "Away hull area (m²)",
    },
    "Ball": {
        "ball_observed": "Ball observed (0/1)",
        "derived_speed_m_s": "Ball speed (m/s)",
        "nearest_slot_distance_m": "Nearest player distance (m)",
        "boundary_distance_m": "Pitch boundary distance (m)",
        "exit_candidate": "Exit candidate (0/1)",
    },
    "Temporal": {
        "temporal_segment": "Tracking segment ID (-1 = outside/cut)",
        "tracking_cut": "Tracking cut (0/1)",
        "w0p4_joint_slowdown_m_s": "0.4 s joint slowdown (m/s)",
        "w1_all_speed_change": "1 s all speed change (m/s)",
        "w1_all_speed_past_coverage": "1 s past player speed coverage",
        "w1_all_speed_future_coverage": "1 s future player speed coverage",
        "w1_joint_slowdown_m_s": "1 s joint slowdown (m/s)",
        "w1_ball_observed_past_fraction": "1 s past ball visibility",
        "w1_ball_observed_future_fraction": "1 s future ball visibility",
        "w1_ball_speed_m_s_change": "1 s ball speed change (m/s)",
        "w1_ball_speed_m_s_past_coverage": "1 s past ball speed coverage",
        "w1_ball_speed_m_s_future_coverage": "1 s future ball speed coverage",
        "w1_ball_nearest_player_distance_m_change": "1 s nearest player distance change (m)",
        "w3_all_speed_change": "3 s all speed change (m/s)",
    },
}


def resolve_feature_path(value):
    """Resolve a generated CSV path without entering the private raw-data tree."""
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path.is_relative_to(ROOT / "data"):
        raise ValueError("Select a generated feature CSV outside the raw data directory.")
    return path


def load_feature_csvs(paths):
    """Join four generated files by exact original frame, reading only chart columns."""
    if set(paths) != set(METRICS):
        raise ValueError("Provide Players, Teams, Ball, and Temporal CSV paths.")
    joined = None
    for source, fields in METRICS.items():
        path = resolve_feature_path(paths[source])
        wanted = set(fields) | {"frame"}
        table = pd.read_csv(path, usecols=lambda column: column in wanted)
        if "frame" not in table or len(table.columns) == 1:
            raise ValueError(f"{source}: expected frame and feature columns in {path}.")
        frame = pd.to_numeric(table["frame"], errors="coerce").to_numpy(dtype=float)
        if (not len(frame) or not np.all(np.isfinite(frame))
                or np.any(frame < 0) or np.any(frame != np.floor(frame))
                or np.any(np.diff(frame) <= 0)):
            raise ValueError(f"{source}: frame IDs must be increasing nonnegative integers.")
        if joined is None:
            joined = pd.DataFrame({"frame": frame.astype(np.int64)})
        elif not np.array_equal(joined["frame"].to_numpy(), frame):
            raise ValueError(f"{source}: frame IDs do not match the other CSVs.")
        for field, label in fields.items():
            if field in table:
                joined[f"{source} · {label}"] = pd.to_numeric(table[field], errors="coerce")
    return joined
