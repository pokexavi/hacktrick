"""The whole match with one model's stoppages, to look for the stoppages it misses."""
from pathlib import Path
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))

import config
import dead_ball_review as review_lib
import missed_review as mr
from media import video_window
from visualizer import parse_atd, parse_events

FPS = config.FPS
MODEL_ROOT = ROOT / getattr(config, "DEAD_ALIVE_ROOT", "cache/dead_alive")
VOTERS = re.compile(r"tcn_bar_final(_nokeeper)?(_s\d+)?_all7$")
WHISTLE_SOURCES = {"preset": "preset", "D30": "D_conc30_prom15_snr15", "D20": "D_conc20_prom12_snr15"}
HELP = """
The other pages start from a stoppage a model predicted. This one starts from what it calls **alive**,
to find the stoppages it misses.

- **Timeline**: both halves, red = Dead, green = alive. Click a stretch to open it from its start. Yellow ticks = candidates
  with score 2 or more; the white tick = the clip now open. *Previous* / *Next* go back to the list.
- **Candidates**: places where something automatic suggests a stoppage the model doesn't have. *score* = how many
  different kinds of sign: *models* (other seeds or variants say Dead; *votes* = the largest share), *p_dead* (the model's
  own p_dead rises but no stoppage comes out; *peak_p*), *whistle* (a whistle in play; `preset` is the strict detector, `D30`
  and `D20` are more permissive, so many of theirs are not the referee), *ball gap* (the ball isn't tracked for ≥ 5 s),
  *tagging* (a Smart Tagging set piece with no stoppage in its window). Pick a row to open it.
  *to_stoppage_s* near 0 is a boundary problem, not a missed stoppage: the filter hides those.
- **Only alive**: steps through the alive stretches, skipping the stoppages (about 58 min; the player goes up to 8×, and keeps the speed from clip to clip).
- **Long alive stretches**: the longest runs without a stoppage.

Everything here is automatic and only says where to look. What you see should feed general changes to the model, not rules tied
to moments of this match (RULES.md).
"""


def timestamp(frame):
    seconds = int(frame / FPS)
    return f"{seconds // 60}:{seconds % 60:02d}"


@st.cache_data(show_spinner="Reading the model...")
def load_frames(path, modified):
    return pd.read_csv(path, usecols=["p_dead", "p_start", "p_stop", "state", "ball_observed"], low_memory=False)


@st.cache_data(show_spinner=False)
def load_state(path, modified):
    return pd.read_csv(path, usecols=["state"], low_memory=False)["state"].to_numpy()


def frames_of(name):
    path = MODEL_ROOT / name / "frames.csv"
    return load_frames(str(path), path.stat().st_mtime_ns)


def state_of(name):
    path = MODEL_ROOT / name / "frames.csv"
    return load_state(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner="Reading Smart Tagging...")
def load_tagging(path, modified):
    return review_lib.smart_tagging_events(parse_events(Path(path).read_text(encoding="utf-8-sig"), FPS))


@st.cache_data(show_spinner="Reading pitch tracking...", max_entries=20)
def load_tracking(raw_path, raw_modified, metadata_path, metadata_modified, start, end):
    """Players and ball of the clip, for the 2D pitch (as on the Dead-ball review page)."""
    metadata = Path(metadata_path).read_text(encoding="utf-8-sig")
    with Path(raw_path).open(encoding="utf-8-sig") as lines:
        return parse_atd(lines, metadata, start, end)


def clip_tracking(start, end):
    """The pitch tracking of [start, end), or None (with a note) when it can't be read."""
    if not (config.ATD_RAW_PATH and config.ATD_METADATA_PATH):
        st.info("Set ATD_RAW_PATH and ATD_METADATA_PATH in visualization_app/config.py for the 2D pitch.")
        return None
    try:
        raw, metadata = ROOT / config.ATD_RAW_PATH, ROOT / config.ATD_METADATA_PATH
        return load_tracking(str(raw), raw.stat().st_mtime_ns, str(metadata), metadata.stat().st_mtime_ns, start, end)
    except (OSError, ValueError, UnicodeError, ET.ParseError) as error:
        st.warning(f"Pitch tracking unavailable: {error}")
        return None


def tagging_events():
    empty = pd.DataFrame(columns=["kind", "team", "minute", "description", "start", "stop"])
    try:
        path = ROOT / config.SMART_TAGGING_PATH
        return review_lib.clip_to_play(load_tagging(str(path), path.stat().st_mtime_ns))
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        st.warning(f"Smart Tagging unavailable: {error}")
        return empty


@st.cache_data(show_spinner=False)
def load_whistles():
    out = {}
    for label, name in WHISTLE_SOURCES.items():
        path = ROOT / "cache" / "audio" / "variants" / f"{name}.csv"
        if path.exists():
            out[label] = pd.read_csv(path)["frame"].to_numpy()
    return out


st.title("Whole match")
st.caption("One model's stoppages over the whole match, and where it may have missed one.")
with st.expander("How to read this page"):
    st.markdown(HELP)

available = sorted(p.parent.name for p in MODEL_ROOT.glob("*/frames.csv"))
if not available:
    st.info(f"No model outputs in {MODEL_ROOT}.")
    st.stop()
newest = [n for n in available if re.fullmatch(r"ens_keeper\d+_all7", n)]
default = newest[-1] if newest else ("ens_final3_all7" if "ens_final3_all7" in available else available[0])

with st.sidebar:
    st.header("Whole match")
    model = st.selectbox("Model", available, index=available.index(default), key="fm_model")
    voters = st.multiselect("Models that can vote", [n for n in available if n != model],
                            default=[n for n in available if VOTERS.fullmatch(n) and n != model], key="fm_voters",
                            help="Their Dead on frames this model calls alive is the 'models' sign.")
    p_threshold = st.slider("p_dead sign above", 0.2, 0.6, 0.35, step=0.05, key="fm_p")
    detectors = st.multiselect("Whistles", list(WHISTLE_SOURCES), default=list(WHISTLE_SOURCES), key="fm_whistles")
    use_tagging = st.checkbox("Smart Tagging sign", value=True, key="fm_tagging")
    seconds = st.slider("Clip length (s)", 30, 300, 120, step=30, key="fm_seconds",
                        help="Longer clips take longer to prepare.")
    margin = st.slider("Seconds shown before a candidate", 0, 30, 8, key="fm_margin")

frames = frames_of(model)
n = len(frames)
play = review_lib.in_play(n)
state = frames["state"].to_numpy()
alive = play & (state == "alive")
dead = play & (state == "D")

signs = (mr.model_signs(alive, np.stack([state_of(v) == "D" for v in voters]) if voters else None)
         + mr.p_dead_signs(alive, frames["p_dead"].to_numpy(), threshold=p_threshold)
         + mr.ball_gap_signs(alive, frames["ball_observed"].to_numpy()))
whistles = load_whistles()
for label in detectors:
    signs += mr.whistle_signs(alive, whistles[label], label) if label in whistles else []
events = tagging_events()
without = None
if use_tagging and not events.empty:
    stops = pd.DataFrame(mr.runs_of(dead), columns=["start", "stop"])
    without = review_lib.without_stoppage(events, stops)
table = mr.candidates(signs, dead, without)
table["time"] = table["start"].map(timestamp)

st.caption(f"**{model}**: {len(mr.runs_of(dead))} stoppages · alive {alive.sum() / FPS / 60:.0f} min · "
           f"dead {dead.sum() / FPS / 60:.0f} min · {len(table)} candidates")


def timeline(current):
    """Both halves: alive/Dead rectangles, candidate ticks, the clip now open."""
    rows, marks = [], []
    for half, (a, b) in enumerate(review_lib.PLAYED, start=1):
        for label, mask in (("alive", alive), ("Dead", dead)):
            rows += [{"half": f"Half {half}", "x": s / FPS / 60, "x2": e / FPS / 60, "state": label, "frame": a + s}
                     for s, e in mr.runs_of(mask[a:b])]
        inside = table[(table["start"] >= a) & (table["start"] < b)]
        marks += [{"half": f"Half {half}", "x": (r.start - a) / FPS / 60, "frame": int(r.start), "score": int(r.score),
                   "time": r.time} for r in inside.itertuples()]
        if current is not None and a <= current < b:
            marks.append({"half": f"Half {half}", "x": (current - a) / FPS / 60, "frame": int(current), "score": -1,
                          "time": "now"})
    click = alt.selection_point(name="pick", fields=["frame"], on="click")
    bars = alt.Chart(pd.DataFrame(rows)).mark_rect().encode(
        x=alt.X("x:Q", title="minutes into the half"), x2="x2:Q", y=alt.Y("half:N", title=None),
        color=alt.Color("state:N", scale=alt.Scale(domain=["alive", "Dead"], range=["#2e7d32", "#c62828"]), legend=None),
        tooltip=["state:N", "frame:Q"]).add_params(click)
    data = pd.DataFrame(marks, columns=["half", "x", "frame", "score", "time"])
    ticks = alt.Chart(data[data["score"] >= 2]).mark_tick(color="#ffd54f", thickness=2, size=22).encode(
        x="x:Q", y="half:N", tooltip=["time:N", "score:Q"])
    now = alt.Chart(data[data["score"] < 0]).mark_tick(color="#ffffff", thickness=3, size=34).encode(x="x:Q", y="half:N")
    return (bars + ticks + now).properties(height=110)


mode = st.radio("Go through", ["Candidates", "Only alive", "Whole match in order", "Long alive stretches"], horizontal=True,
                key="fm_mode")
items = None
if mode == "Candidates":
    a, b, c = st.columns(3)
    min_score = a.slider("Minimum score", 1, 5, 2, key="fm_score")
    min_far = b.slider("At least this far from a stoppage (s)", 0.0, 15.0, 3.0, step=1.0, key="fm_far")
    whistle_only = c.checkbox("Only with a preset whistle", key="fm_preset")
    shown = table[(table["score"] >= min_score) & (table["to_stoppage_s"] >= min_far)]
    if whistle_only:
        shown = shown[shown["whistles"].str.contains("preset")]
    shown = shown.reset_index(drop=True)
    st.caption(f"{len(shown)} of {len(table)} candidates")
    if shown.empty:
        st.info("No candidates for this choice.")
        st.stop()
    pick = st.dataframe(shown[["time", "seconds", "score", "signs", "votes", "peak_p", "whistles", "tagged", "to_stoppage_s",
                               "start", "stop"]],
                        hide_index=True, width="stretch", height=280, on_select="rerun", selection_mode="single-row",
                        key=f"fm_table_{model}_{min_score}_{min_far}_{whistle_only}")
    rows = pick.selection.rows
    if rows and rows[0] != st.session_state.get("fm_row"):      # a new row picked (the selection stays set)
        st.session_state["fm_index"] = rows[0]
        st.session_state.pop("fm_jump", None)
    st.session_state["fm_row"] = rows[0] if rows else None
    items = [(max(int(r.start) - margin * FPS, 0), int(r.stop) + margin * FPS) for r in shown.itertuples()]
elif mode == "Only alive":
    items = mr.pieces(alive, seconds)
elif mode == "Whole match in order":
    items = mr.pieces(play, seconds)
else:
    longest = mr.long_alive(alive, min_s=st.slider("At least (s)", 30, 120, 60, step=10, key="fm_long"))
    longest["time"] = longest["start"].map(timestamp)
    if longest.empty:
        st.info("No alive stretch that long.")
        st.stop()
    st.dataframe(longest[["time", "seconds", "start", "stop"]], hide_index=True, width="stretch", height=200)
    items = [(s, min(s + seconds * FPS, int(row.stop))) for row in longest.itertuples()
             for s in range(int(row.start), int(row.stop), seconds * FPS)]

key = f"fm_index_{mode}_{seconds}_{model}"
index_key = "fm_index" if mode == "Candidates" else key
index = min(int(st.session_state.get(index_key, 0)), len(items) - 1)


def go(target):
    st.session_state[index_key] = target
    st.session_state.pop("fm_jump", None)


prev_col, label_col, next_col, jump_col = st.columns([1, 3, 1, 2])
prev_col.button("◀ Previous", on_click=go, args=(index - 1,), disabled=index == 0)
next_col.button("Next ▶", on_click=go, args=(index + 1,), disabled=index >= len(items) - 1)
start, stop = items[index]
half_start, half_end = review_lib.half_bounds(start)
start, stop = max(start, half_start), min(stop, half_end, start + 300 * FPS)
label_col.markdown(f"**{index + 1} / {len(items)}** · {timestamp(start)}–{timestamp(stop)} · frames {start}–{stop}")
minute = jump_col.number_input("Open at minute (video time)", min_value=0.0, value=None, step=0.5, key="fm_minute",
                               placeholder="e.g. 50.9")
opened = int(minute * 60 * FPS) if minute is not None else st.session_state.get("fm_jump")
if opened is not None:
    half_start, half_end = review_lib.half_bounds(opened)
    start = max(opened, half_start)
    stop = min(start + seconds * FPS, half_end)
    st.caption("Showing the minute typed above; clear it to go back to the list." if minute is not None else
               f"Showing the stretch clicked in the timeline ({timestamp(start)}). Previous / Next go back to the list.")

chosen = st.altair_chart(timeline(start), on_select="rerun", width="stretch", key=f"fm_timeline_{model}")
clicked = chosen.selection.get("pick", []) if hasattr(chosen, "selection") else []
click = int(clicked[0]["frame"]) if clicked and "frame" in clicked[0] else None
if click is not None and click != st.session_state.get("fm_click"):     # a new click (the selection stays set)
    st.session_state["fm_click"] = click
    st.session_state["fm_jump"] = click
    st.rerun()

try:
    path = ROOT / config.TACTICAL_VIDEO_PATH
    encoded = video_window(str(path), path.stat().st_mtime_ns, FPS, start, stop)
except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as error:
    st.warning(f"Could not prepare the video: {error}")
    st.stop()


def rounded(values):
    return [None if not np.isfinite(v) else round(float(v), 3) for v in values]


span = slice(start, stop)
masked = np.where(play, state, "X")
ball = frames["ball_observed"].to_numpy(dtype=float)[span]
in_clip = events[(events["stop"] > start) & (events["start"] < stop)]
preset = whistles.get("preset", np.array([], dtype=int))
payload = {
    "clipStart": start, "clipEnd": stop, "fps": FPS, "selStart": start, "selStop": stop,
    "gt": "X" * (stop - start), "reference": 0,
    "ball": [None if not np.isfinite(b) else int(b) for b in ball],
    "models": [{"name": model, "colour": "#ffffff", "state": [str(s) for s in masked[span]],
                "p": rounded(frames["p_dead"].to_numpy(dtype=float)[span]),
                "pStart": rounded(frames["p_start"].to_numpy(dtype=float)[span]),
                "pStop": rounded(frames["p_stop"].to_numpy(dtype=float)[span]), "decisions": None}],
    "events": [{"start": int(e.start), "stop": int(e.stop), "label": f"{e.kind} {e.minute}", "colour": "#ffb74d"}
               for e in in_clip.itertuples()],
    "tracking": clip_tracking(start, stop),
    "whistles": [[int(f), int(f) + FPS // 2] for f in preset if start <= f < stop],
    "spec": None,
}
player = (APP_DIR / "dead_ball_player.html").read_text(encoding="utf-8")
player = player.replace("__VIDEO_BASE64__", encoded).replace("__PAYLOAD__", json.dumps(payload).replace("<", "\\u003c"))
components.html(player, height=1170, scrolling=True)
