"""The corruptions of dead_alive_detection/augment, as the viewer offers and applies them.

The same code and the same draws as training (augment/views.py): copy 0 is the scored view of a
match, copy 1 + k its training copy k. Only models already in the cache are used, and the BAR-ATH
library is never built from here: an option whose file is missing is simply not offered.
"""
import json

import streamlit as st

from config import CACHE_DIR
from dead_alive_detection import config as experiment
from dead_alive_detection.augment.views import Views
from external import schema
from shared import tracking_from_match

PLAYERS_MODEL = CACHE_DIR / "players_learned" / "BAR-ATH.json"
BALL_MODEL = CACHE_DIR / "ball_loss" / "BAR-ATH.json"
SWAPS_MODEL = CACHE_DIR / "player_swaps" / "BAR-ATH.json"
LIBRARY = CACHE_DIR / "dead_alive_dfl" / "bar_motion" / "library.npz"
FIT_PLAYERS = ("python -m dead_alive_detection.augment.players.fit_learned "
               "--match data/processed/BAR-ATH.parquet")
FIT_BALL = "python -m dead_alive_detection.augment.ball.fit_hidden --match data/processed/BAR-ATH.parquet"
BUILD_LIBRARY = "python -m dead_alive_detection.augment.dead_ball.bar"
FIT_SWAPS = ("python -m dead_alive_detection.augment.players.fit_swaps "
             "--match data/processed/BAR-ATH.parquet")

# {corruption: {option: (what it does, file it needs, how to make that file)}}
OPTIONS = {
    "swaps": {
        "none": ("Every track follows one player, as DFL has them.", None, None),
        "bar": ("Tracks swap players progressively, within a team or across teams, at BAR-ATH's "
                "rate and with the shapes of its swaps: half of them in dead ball, half in play "
                "(learnt from BAR).", SWAPS_MODEL, FIT_SWAPS),
    },
    "players": {
        "clean": ("Every player tracked, under a fixed identity, as DFL has them.", None, None),
        "fragments": ("Tracks cut at random into pieces in anonymous slots, plus duplicates. Rates "
                      "and piece lengths from BAR-ATH.", PLAYERS_MODEL, FIT_PLAYERS),
        "learned": ("The same slots and duplicates, but players vanish and come back in the "
                    "situations they do on BAR-ATH (learnt model).", PLAYERS_MODEL, FIT_PLAYERS),
    },
    "ball": {
        "clean": ("The ball is always seen, as DFL has it.", None, None),
        "hidden": ("The ball is hidden the way BAR-ATH's tracker loses it (learnt model).",
                   BALL_MODEL, FIT_BALL),
    },
    "dead_ball": {
        "frozen": ("The ball stays still through every stoppage, as DFL has it.", None, None),
        "uniform": ("A made-up path through every stoppage, at one median speed.", None, None),
        "bar": ("Real pieces of BAR-ATH's dead-ball motion through every stoppage.",
                LIBRARY, BUILD_LIBRARY),
    },
}
RECIPE = "viewer"


def available(corruption):
    """The options of a corruption whose file is there."""
    return [name for name, (_, needs, _) in OPTIONS[corruption].items()
            if needs is None or needs.is_file()]


def missing(corruption):
    """{option: how to make its file} of the options that are not available."""
    return {name: how for name, (_, needs, how) in OPTIONS[corruption].items()
            if needs is not None and not needs.is_file()}


def bar_hidden_share():
    """The share of steps BAR-ATH hides the ball on, stored with the ball model."""
    with open(BALL_MODEL) as handle:
        return json.load(handle)["hidden_frac"]


def _config(players, ball, dead_ball, swaps="none"):
    cfg = experiment.load(experiment.BASE)
    aug = cfg["augment"]
    aug["recipes"] = {RECIPE: {"players": players, "ball": ball, "dead_ball": dead_ball,
                           "swaps": swaps}}
    aug["train"]["recipe"] = RECIPE
    aug["eval"]["conditions"] = [RECIPE]
    aug["eval"]["primary"] = RECIPE
    aug["players"]["fragments"]["profile"] = str(PLAYERS_MODEL)   # the same BAR-ATH aggregates
    aug["players"]["learned"]["model"] = str(PLAYERS_MODEL)
    aug["ball"]["model"] = str(BALL_MODEL)
    aug["swaps"]["model"] = str(SWAPS_MODEL)
    aug["dead_ball"]["bar"]["library"] = str(LIBRARY)
    return cfg


def _stamp(path):
    return path.stat().st_mtime_ns if path.is_file() else 0


@st.cache_resource(show_spinner="Applying the corruptions to this match...", max_entries=3)
def _corrupted(path, modified, players, ball, dead_ball, hidden_frac, copy, models_modified,
               swaps="none"):
    names = [p.replace("\\", "/") for p in schema.processed_paths("dfl")]
    index = names.index(path.replace("\\", "/"))
    match, report = Views(_config(players, ball, dead_ball, swaps)).corrupted_match(
        schema.load_match(path), index, copy, RECIPE, hidden_frac)
    report = dict(report, swaps=match.meta.get("swaps", []))
    label = (f"Corrupted DFL: swaps {swaps}, players {players}, ball {ball}, "
             f"dead ball {dead_ball}, copy {copy}")
    return tracking_from_match(match, source_format=label), report


def corrupted_tracking(source, choice):
    """(the viewer's tracking dict, report of the ball hiding) of a DFL match under these choices."""
    path = source["tracking"]
    return _corrupted(str(path), _stamp(path), choice["players"], choice["ball"],
                      choice["dead_ball"], choice["hidden_frac"], choice["copy"],
                      (_stamp(PLAYERS_MODEL), _stamp(BALL_MODEL), _stamp(LIBRARY),
                       _stamp(SWAPS_MODEL)), choice.get("swaps", "none"))
