"""Review and compare Dead/Alive models: seconds before a stoppage, the stoppage, and after."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

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
import whistle_review as wr
from feature_explorer import resolve_feature_path
from media import video_window
from visualizer import parse_atd, parse_events

COLOURS = ["#ffffff", "#ffb74d", "#4dd0e1", "#ba68c8", "#aed581", "#f06292"]
BASE_COLUMNS = ("frame", "p_dead_frame", "p_dead", "state", "ball_observed", "p_start", "p_stop")
EVENT_COLOURS = {"goal": "#f1c40f", "corner": "#ff9800", "throw_in": "#5dade2", "goal_kick": "#16a085",
                 "free_kick": "#8e44ad", "kickoff": "#ffffff", "kickoff_after_goal": "#ffffff"}
HELP = """
**What a model gives, every 0.2 s** (the TCN's three outputs, plus the type for the typed ones):

- `p_dead`: is the ball dead now? Reliable in the middle of a stoppage, but it ramps up and down over
  a few seconds at the edges, so a 0.5 threshold can put the edge 2-3 s off.
- `p_start` / `p_stop`: trained to **peak at the instant** a stoppage begins / play restarts
  (±0.4 s). They are low everywhere else, so a peak of 0.3 already means "an edge here".
- `p_<class>` (typed models): which kind of stoppage it is (whistle, throw-in, goal kick, corner,
  goal, incident, other). The type weight (0.5 or 0.1) is how much training cared about it.

**The decoder (HSMM)** turns these into stoppages: it follows `p_dead`, prefers durations seen in
the Bundesliga, and gets a bonus for cutting where `p_start` / `p_stop` peak (`w_start`, `w_stop`).
In the player, *Heads of* draws one model's `p_start`, `p_stop` and its decoded edges (vertical
lines), so you can see whether an edge follows a peak or only the `p_dead` ramp.

**Models disagree** splits the runs where models differ: *boundary shifted* (all see the stoppage,
they cut it at different places), *gap inside a stoppage* (one model splits a stoppage the others
keep whole), *stoppage seen by some models*.

**Why the decoder cut here** (`decisions.csv`, from `python -m dead_alive_detection.decoding.explain`):
the HSMM's score is a sum, so each kept segment can be compared with the alternative of absorbing
it into its neighbours (a stoppage dropped, or a gap filled so two stoppages become one). For
keeping it: *prior* (duration priors; < 0 = they would rather absorb it), *TCN* (p_dead),
*boundary* (start / stop heads), and *margin* = their sum. A small margin is a close call: the
only places where tuning the decoder can change anything. In the player the margins are written
over each segment of the *Heads of* model (red below 10).

**Whistles** (the broadcast detector's preset, `cache/audio/BAR-ATH_broadcast_whistles_preset.csv`,
what the whistle models read): green marks in the *whistles* row and green lines over the curves.
*Detected whistles* steps through all of them; the kind says where each falls against the reference
model. With a whistle model as reference (`whistle.json` in its folder) the rows also say what the
whistle changes against the same model without whistles (its counterfactual, e.g.
`ens_whistle3_nowhistle`: add it to the comparison to see both), and *Whistle changes* lists every
stoppage that differs between the two. *Sound* plays the tactical or the broadcast audio with the
clip; *Spectrogram* draws the broadcast audio (1.5-5.5 kHz) under the rows: a referee's whistle is a
short bright horizontal line at ~3.8 kHz. In *Why the decoder cut here*, `whistle` is the part of the
margin that comes from the whistles (margin minus `margin_no_whistle`, the same segment scored
without them); `margin_no_whistle` < 0: without the whistle the decoder would not keep the segment.

**Smart Tagging events**: the set pieces and goals of Metrica's automatic tagging (~35 s windows
around the restart; recall is partial). Algorithmic, so allowed (FAQ.md); not ground truth.
*Smart Tagging disagree* keeps the events whose window has no stoppage at all of the reference model.

**Rules (RULES.md, 2026-10-23).** No hand labels and no human-produced data about this game (event
data from any source, corrected tracking, annotations), and the public window is for checking the
pipeline, never for choosing or tuning a model. Watching the video to understand and debug is fine.
"""


def configured_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def timestamp(frame, fps):
    minutes, seconds = divmod(frame / fps, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def runs_of(mask, offset=0):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return [(offset + int(a), offset + int(b))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


@st.cache_data(show_spinner="Reading model outputs...")
def load_model(folder, modified):
    frames = pd.read_csv(Path(folder) / "frames.csv", low_memory=False)
    stops = pd.read_csv(Path(folder) / "intervals.csv")
    return frames, stops


@st.cache_data(show_spinner="Reading decoder decisions...")
def load_decisions(path, modified):
    return pd.read_csv(path)


def decisions_of(folder):
    """decisions.csv of a model folder (why each segment was kept), or None (typed or older)."""
    path = Path(folder) / "decisions.csv"
    return load_decisions(str(path), path.stat().st_mtime_ns) if path.exists() else None


@st.cache_data(show_spinner="Scoring models...")
def model_scores(folders, modified):
    """Public-window and whole-match aggregates from dead_alive_detection.evaluation.bar_window."""
    from dead_alive_detection.evaluation import bar_window as compare
    gt = compare.labels(compare.GT_PATH)
    table = {}
    for folder in folders:
        model = compare.load(folder)
        table[model["name"]] = {**compare.window_metrics(model, gt), **compare.match_metrics(model)}
    return pd.DataFrame(table)


@st.cache_data(show_spinner="Reading pitch tracking...")
def load_tracking(raw_path, raw_modified, metadata_path, metadata_modified, start, end):
    """Players and ball of the clip window, for the 2D pitch (as on the Events page)."""
    metadata = Path(metadata_path).read_text(encoding="utf-8-sig")
    with Path(raw_path).open(encoding="utf-8-sig") as lines:
        return parse_atd(lines, metadata, start, end)


def clip_tracking(start, end):
    """The pitch tracking of [start, end), or None (with a note) when it is not configured."""
    if not (config.ATD_RAW_PATH and config.ATD_METADATA_PATH):
        st.info("Set ATD_RAW_PATH and ATD_METADATA_PATH in visualization_app/config.py for the 2D pitch.")
        return None
    try:
        raw = configured_path(config.ATD_RAW_PATH)
        metadata = configured_path(config.ATD_METADATA_PATH)
        return load_tracking(str(raw), raw.stat().st_mtime_ns,
                             str(metadata), metadata.stat().st_mtime_ns, start, end)
    except (OSError, ValueError, UnicodeError, ET.ParseError) as error:
        st.warning(f"Pitch tracking unavailable: {error}")
        return None


@st.cache_data
def load_gt(path, modified):
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()


@st.cache_data(show_spinner="Reading Smart Tagging...")
def load_tagging_events(path, modified, fps):
    """Smart Tagging set pieces and goals, as frame windows."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return review_lib.smart_tagging_events(parse_events(text, fps), fps)


@st.cache_data(show_spinner=False)
def load_whistles(path, modified):
    table = pd.read_csv(path)
    table["stop"] = table["frame"] + np.maximum(1, np.round(table["duration_s"] * config.FPS)).astype(int)
    return table


def detected_whistles():
    """The preset whistles (frame, stop, ...) or None."""
    path = configured_path(getattr(config, "WHISTLES_PATH", "cache/audio/BAR-ATH_broadcast_whistles_preset.csv"))
    return load_whistles(str(path), path.stat().st_mtime_ns) if path.exists() else None


@st.cache_data(show_spinner="Preparing the clip with sound...", max_entries=10)
def clip_with_sound(image, image_s, sound, sound_s, seconds, highpass):
    return wr.av_clip(image, image_s, sound, sound_s, seconds, highpass)


@st.cache_data(show_spinner=False, max_entries=10)
def clip_spectrogram(wav, start_s, seconds):
    return wr.spectrogram_png(wav, start_s, seconds, max_cols=1600)


def tagging_events(fps):
    """The Smart Tagging events, or an empty table (with a note) when they cannot be read."""
    empty = pd.DataFrame(columns=["kind", "team", "minute", "description", "start", "stop"])
    if not config.SMART_TAGGING_PATH:
        return empty
    try:
        path = configured_path(config.SMART_TAGGING_PATH)
        return load_tagging_events(str(path), path.stat().st_mtime_ns, fps)
    except (OSError, ValueError, KeyError, UnicodeError, ET.ParseError) as error:
        st.warning(f"Smart Tagging unavailable: {error}")
        return empty


st.title("Dead-ball review")
st.caption("Compare Dead/Alive models on video, with the Smart Tagging set pieces as landmarks. "
           "Only the halves as played are shown: from each kickoff to the last graded frame "
           "(frames 11185-90651 and 113742-190197; the graded frames before the second-half "
           "kickoff are D by the rules). "
           "Generate model outputs with `python -m dead_alive_detection.run --config ...` "
           "or `python -m dead_alive_detection.predict --run results/<run> --name <name>`.")
with st.expander("How to read this page (outputs, start/stop heads, decoder, events, rules)"):
    st.markdown(HELP)

root_text = st.text_input("Folder with model outputs", getattr(config, "DEAD_ALIVE_ROOT", "cache/dead_alive"))
try:
    model_root = resolve_feature_path(root_text)
    available = sorted(p.parent.name for p in model_root.glob("*/frames.csv"))
except (OSError, ValueError) as error:
    st.error(f"Could not list model outputs: {error}")
    st.stop()
if not available:
    st.info(f"No model outputs found in {model_root}. Run `python -m dead_alive_detection.run` first.")
    st.stop()
default = [name for name in getattr(config, "DEAD_ALIVE_DEFAULT", []) if name in available] \
    or available[:3]
chosen = st.multiselect("Models to compare", available, default=default, max_selections=len(COLOURS))
if not chosen:
    st.stop()
reference = st.selectbox("Reference model (list of stoppages, video border, evidence)", chosen)

try:
    models = {}
    for name in chosen:
        folder = model_root / name
        modified = tuple((folder / f).stat().st_mtime_ns for f in ("frames.csv", "intervals.csv"))
        models[name] = load_model(str(folder), modified)
    gt_path = configured_path(config.GT_PATH)
    gt = load_gt(str(gt_path), gt_path.stat().st_mtime_ns)
except (OSError, ValueError) as error:
    st.error(f"Could not read the model outputs: {error}")
    st.stop()
if any(len(frames) != len(gt) for frames, _ in models.values()):
    st.error("A model output and the GT have different lengths.")
    st.stop()

fps = config.FPS
# Only the halves as played (kickoff to last graded frame): everything else is left out.
play = review_lib.in_play(len(gt))
stops_of = {name: review_lib.clip_to_play(stops) for name, (_, stops) in models.items()}
explained = {name: review_lib.clip_to_play(decisions_of(model_root / name)) for name in chosen}
typed = [name for name in chosen if "type" in stops_of[name]]
events = review_lib.clip_to_play(tagging_events(fps))
whistles = detected_whistles()
if whistles is not None:
    whistles = whistles[review_lib.in_play(len(gt))[np.clip(whistles["frame"], 0, len(gt) - 1)]].reset_index(drop=True)
ref_info = wr.whistle_info(model_root / reference)
ref_effects = wr.influence(model_root / reference) if ref_info else None

with st.expander("The models", expanded=True):
    st.dataframe(pd.DataFrame([review_lib.describe_model(model_root / name) for name in chosen]),
                 width="stretch", hide_index=True)
with st.expander("Pipeline check on the public window (4 GT dead balls; by the rules never to choose or tune)"):
    folders = tuple(str(model_root / name) for name in chosen)
    stamp = tuple((model_root / name / "frames.csv").stat().st_mtime_ns for name in chosen)
    scores = model_scores(folders, stamp)
    # Mixed numbers and "4/4"-style counts: format as text so the table renders as is.
    st.dataframe(scores.map(lambda v: f"{v:.3g}" if isinstance(v, (int, float, np.floating, np.integer))
                            and not isinstance(v, bool) else str(v)), width="stretch")

ref_frames = models[reference][0]
ref_stops = stops_of[reference]
states = {name: np.where(play, frames["state"].to_numpy(), "X") for name, (frames, _) in models.items()}
window = np.flatnonzero(gt != "X")
tagging_disagree = f"Smart Tagging disagree ({reference})"
modes = ["Smart Tagging events", f"Stoppages of {reference}", tagging_disagree]
if len(chosen) > 1:
    modes.append("Models disagree (whole match)")
if explained[reference] is not None:
    modes.append(f"Close calls of the decoder ({reference})")
if whistles is not None and len(whistles):
    modes.append("Detected whistles")
if (model_root / reference / "whistle_changes.csv").exists():
    modes.append(f"Whistle changes ({reference})")
modes += ["GT dead balls (public window)", f"{reference} vs GT disagreements (public window)"]
review = st.radio("Review", modes, horizontal=True)
tagging_review = review in ("Smart Tagging events", tagging_disagree)

if tagging_review:
    if events.empty:
        st.info("No Smart Tagging events: set SMART_TAGGING_PATH in visualization_app/config.py.")
        st.stop()
    kinds = sorted(events["kind"].unique())
    kinds = st.multiselect("Event kinds", kinds, default=kinds)
    table = events[events["kind"].isin(kinds)].copy()
    if table.empty:
        st.info("No events for this choice.")
        st.stop()
    st.markdown("**Per kind and model**: share of events with a predicted stoppage in the window · "
                "mean number of stoppages there (above 1: cut in pieces, or several stoppages in the window)")
    st.dataframe(review_lib.coverage(table, stops_of), width="stretch", hide_index=True)
    if review == tagging_disagree:
        total = len(table)
        table = review_lib.without_stoppage(table, stops_of[reference])
        st.caption(f"{len(table)} of {total} events have no stoppage of {reference} in their window: "
                   "not one frame of Dead.")
    for name in chosen:
        table[name] = [review_lib.summarise(stops_of[name], a, b)
                       for a, b in zip(table["start"], table["stop"])]
elif review == modes[1]:
    table = ref_stops[["start", "stop"]].assign(
        kind=ref_stops["type"].fillna("predicted") if "type" in ref_stops else "predicted")
elif review == "Models disagree (whole match)":
    dead = np.vstack([s == "D" for s in states.values()])
    graded = states[reference] != "X"
    split = graded & dead.any(axis=0) & ~dead.all(axis=0)
    table = pd.DataFrame([{"start": a, "stop": b, "kind": review_lib.disagreement_kind(dead, a, b)}
                          for a, b in runs_of(split) if b - a >= fps], columns=["start", "stop", "kind"])
    names = ", ".join(n for n in chosen)
    st.caption(f"Runs of ≥ 1 s where {names} do not all agree. Each row says which models call it dead.")
    table["dead in"] = [", ".join(n for k, n in enumerate(chosen) if dead[k, a:b].mean() > 0.5)
                        for a, b in zip(table["start"], table["stop"])]
elif review.startswith("Close calls"):
    limit = st.number_input("Margin below", min_value=0.0, value=10.0, step=5.0,
                            help="How much the decoder's choice beats absorbing the segment into its neighbours.")
    close = explained[reference]
    close = close[close["margin"] < limit]
    table = close.assign(kind=np.where(close["state"] == "D", "keep stoppage", "keep alive gap"))[
        ["start", "stop", "kind", "prior", "tcn", "boundary", "margin"]]
elif review == "Detected whistles":
    category, _ = wr.categorise(whistles["frame"].to_numpy(), ref_stops)
    table = pd.DataFrame({"start": whistles["frame"], "stop": whistles["stop"],
                          "kind": [f"whistle {c}" for c in category]})
    if ref_info:
        st.caption(f"Kind: where each whistle falls against {reference}. 'effect': what it changes against "
                   f"{ref_info['counterfactual']} (the same model without whistles).")
    if ref_effects is not None:
        k = wr.match_rows(table["start"].to_numpy(), ref_effects)
        table["without whistle"] = [ref_effects["category_no_whistle"].iloc[i] if i >= 0 else "" for i in k]
        table["effect"] = [ref_effects["effect"].iloc[i] if i >= 0 else "" for i in k]
        table["decisive"] = [bool(ref_effects["decisive"].iloc[i]) if i >= 0 else False for i in k]
elif review.startswith("Whistle changes"):
    changes = pd.read_csv(model_root / reference / "whistle_changes.csv")
    table = changes.rename(columns={"change": "kind"})[["start", "stop", "kind", "start_shift_s", "end_shift_s"]]
    st.caption(f"Every stoppage of {reference} that differs from {ref_info['counterfactual'] if ref_info else 'its counterfactual'} "
               "(the same model without whistles). 'whistle s': from the moved edges to the nearest whistle.")
elif review == "GT dead balls (public window)":
    table = pd.DataFrame([{"start": a, "stop": b, "kind": "GT dead ball"}
                          for a, b in runs_of(gt[window] == "D", window[0])])
else:
    ref_state = states[reference]
    rows = []
    for kind, mask in (("model Dead, GT alive", (ref_state == "D") & np.isin(gt, ["H", "A"])),
                       ("model alive, GT Dead", (ref_state == "alive") & (gt == "D"))):
        rows += [{"start": a, "stop": b, "kind": kind} for a, b in runs_of(mask[window], window[0])
                 if b - a >= fps // 2]
    table = pd.DataFrame(rows, columns=["start", "stop", "kind"]).sort_values("start")
if table.empty:
    st.info("Nothing to review for this choice.")
    st.stop()

table = table.reset_index(drop=True)
table["seconds"] = (table["stop"] - table["start"]) / fps
for name in typed:
    table[f"type {name}"] = [", ".join(review_lib.types_in(stops_of[name], a, b))
                             for a, b in zip(table["start"], table["stop"])]
for name, (frames, _) in models.items():
    p = frames["p_dead"].to_numpy(dtype=float)
    table[f"p {name}"] = [round(float(np.nanmean(p[a:b])), 2) if np.isfinite(p[a:b]).any() else None
                          for a, b in zip(table["start"], table["stop"])]
table["GT Dead share"] = [round(float((gt[a:b] == "D").mean()), 2) if (gt[a:b] != "X").any() else None
                          for a, b in zip(table["start"], table["stop"])]
if whistles is not None and len(whistles):
    marks = whistles["frame"].to_numpy()
    if review.startswith("Whistle changes"):
        table["whistle s"] = changes["whistle_s"].to_numpy()
    else:   # seconds from the row's start to the nearest whistle (- before it)
        gap = marks[None, :] - table["start"].to_numpy()[:, None]
        table["whistle s"] = np.round(gap[np.arange(len(table)), np.abs(gap).argmin(1)] / fps, 1)

filter_a, filter_b, filter_c, filter_d = st.columns(4)
with filter_a:
    row_kinds = sorted(table["kind"].unique())
    keep_kinds = st.multiselect("Row kinds", row_kinds, default=row_kinds) if len(row_kinds) > 1 else row_kinds
with filter_b:
    predicted = sorted({k for name in typed for k in stops_of[name]["type"].dropna().astype(str)})
    keep_types = st.multiselect("Predicted type (any typed model)", predicted,
                                help="Keep the rows where a chosen typed model predicts a stoppage of "
                                     "these types. Empty: no filter.") if predicted else []
with filter_c:
    min_seconds = st.number_input("Minimum duration (s)", min_value=0.0, value=0.0, step=1.0)
with filter_d:
    order = st.selectbox("Order", ["Chronological", "Least confident first", "Longest first", "Random sample"])
table = table[table["kind"].isin(keep_kinds) & (table["seconds"] >= min_seconds)]
if keep_types:
    hit = np.zeros(len(table), dtype=bool)
    for name in typed:
        hit |= table[f"type {name}"].str.split(", ").apply(lambda ks: bool(set(ks) & set(keep_types))).to_numpy()
    table = table[hit]
if "whistle s" in table and st.checkbox("Only rows within 3 s of a whistle"):
    table = table[table["whistle s"].abs() <= 3]
if order == "Least confident first":
    table = table.sort_values(f"p {reference}")
elif order == "Longest first":
    table = table.sort_values("seconds", ascending=False)
elif order == "Random sample":
    seed = st.number_input("Random seed", min_value=0, value=0, step=1, help="Change it to draw another sample.")
    table = table.sample(frac=1, random_state=int(seed))
table = table.reset_index(drop=True)
if table.empty:
    st.info("No intervals match the filters.")
    st.stop()

shown = table.assign(**{"start time": [timestamp(f, fps) for f in table["start"]],
                        "last frame": table["stop"] - 1})
if tagging_review:
    columns = ["kind", "minute", "description"] + chosen + ["start", "start time", "seconds"]
else:
    columns = ["kind", "start", "last frame", "start time", "seconds"]
    columns += ["dead in"] if "dead in" in shown else []
    columns += [c for c in ("prior", "tcn", "boundary", "margin") if c in shown]
    columns += [c for c in ("without whistle", "effect", "decisive", "start_shift_s", "end_shift_s", "whistle s")
                if c in shown]
    columns += [f"type {n}" for n in typed] + [f"p {n}" for n in chosen] + ["GT Dead share"]
event = st.dataframe(shown[columns], width="stretch", hide_index=True, on_select="rerun",
                     selection_mode="single-row")
index = event.selection.rows[0] if event.selection.rows else 0
row = table.iloc[index]
start, stop = int(row["start"]), int(row["stop"])

title = row["description"] if tagging_review else row["kind"]
st.subheader(f"{title} · frames {start}–{stop - 1} · {row['seconds']:.1f} s")
control_a, control_b, control_c = st.columns(3)
with control_a:
    before = st.slider("Seconds before", 0, 60, 5)
with control_b:
    after = st.slider("Seconds after", 0, 120, 30 if tagging_review else 5)
with control_c:
    max_clip = st.slider("Longest clip (s)", 30, 300, 120, step=30,
                         help="Long clips take longer to prepare.")
half_start, half_end = review_lib.half_bounds(start)
clip_start = max(half_start, start - before * fps)
clip_end = min(half_end, stop + after * fps)
if clip_end - clip_start > max_clip * fps:
    clip_end = clip_start + max_clip * fps
    st.caption(f"Long interval: showing the first {max_clip} s of the clip.")
WAV = ROOT / "cache" / "audio" / "BAR-ATH_broadcast_mono16k.wav"
whistle_in_clip = whistles is not None and bool(((whistles["stop"] > clip_start) & (whistles["frame"] < clip_end)).any())
sound_a, sound_b, sound_c = st.columns(3)
with sound_a:
    sound = st.radio("Sound", ["none", "tactical", "broadcast"], horizontal=True,
                     help="The broadcast's sound is aligned to the tactical clock (audio/align.py).")
with sound_b:
    highpass = st.checkbox("Only the whistle band (high-pass 2.5 kHz)", disabled=sound == "none")
with sound_c:
    show_spec = st.checkbox("Spectrogram of the broadcast audio", value=whistle_in_clip, disabled=not WAV.exists(),
                            help="On by default when a detected whistle falls in the clip.")
b0 = float(wr.align.to_broadcast([clip_start])[0])
b1 = float(wr.align.to_broadcast([clip_end - 1])[0])
broadcast_ok = np.isfinite(b0) and np.isfinite(b1) and b1 > b0
if (sound == "broadcast" or show_spec) and not broadcast_ok:
    st.caption("The broadcast does not cover this clip (before it starts, or across halftime): no broadcast sound "
               "or spectrogram.")

terms = [c for c in ref_frames.columns
         if c not in BASE_COLUMNS and c != "dead_type" and not c.startswith("p_")]
if terms:
    spans = {"before": (clip_start, start), "during": (start, stop), "after": (stop, clip_end)}
    evidence = pd.DataFrame({name: ref_frames[terms].iloc[a:b].mean() for name, (a, b) in spans.items() if b > a})
    with st.expander(f"Evidence behind {reference} (mean log-odds; > 0 pushes towards Dead)"):
        st.dataframe(evidence.round(2), width="stretch")

if not config.TACTICAL_VIDEO_PATH:
    st.info("Set TACTICAL_VIDEO_PATH in visualization_app/config.py to watch clips.")
    st.stop()
try:
    video_path = configured_path(config.TACTICAL_VIDEO_PATH)
    broadcast_path = configured_path(getattr(config, "BROADCAST_VIDEO_PATH", "data/broadcast/BAR-ATH_broadcast.mp4"))
    if sound == "tactical" or (sound == "broadcast" and broadcast_ok):
        source, source_s = (video_path, clip_start / fps) if sound == "tactical" else (broadcast_path, b0)
        encoded = clip_with_sound(str(video_path), clip_start / fps, str(source), source_s,
                                  (clip_end - clip_start) / fps, 2500 if highpass else None)
    else:
        encoded = video_window(str(video_path), video_path.stat().st_mtime_ns, fps, clip_start, clip_end)
except (OSError, ValueError, subprocess.CalledProcessError) as error:
    st.warning(f"Could not prepare the configured video: {error}")
    st.stop()


def display_state(name):
    """Model state per frame, with Dead frames of a typed model shown by their type (e.g. corner)."""
    frames = models[name][0]
    state = states[name]
    if "dead_type" not in frames:
        return state
    kind = frames["dead_type"].fillna("").to_numpy()
    return np.where((state == "D") & (kind != "") & (kind != "generic"), kind, state)


def rounded(values):
    return [None if not np.isfinite(value) else round(float(value), 3) for value in values]


def clip_decisions(name):
    table = explained.get(name)
    if table is None:
        return None
    part = table[(table["stop"] > clip_start) & (table["start"] < clip_end)]
    return [{"start": int(r.start), "stop": int(r.stop), "margin": float(r.margin),
             "whistle": float(r.whistle) if "whistle" in part and np.isfinite(r.whistle) else None}
            for r in part.itertuples()]


def column(frames, name):
    if name not in frames:
        return None
    return rounded(np.where(play[span], frames[name].to_numpy(dtype=float)[span], np.nan))


span = slice(clip_start, clip_end)
ball = ref_frames["ball_observed"].to_numpy(dtype=float)[span]
in_clip = events[(events["stop"] > clip_start) & (events["start"] < clip_end)]
payload = {
    "clipStart": clip_start, "clipEnd": clip_end, "fps": fps, "selStart": start, "selStop": stop,
    "gt": "".join(gt[span]), "reference": chosen.index(reference),
    "ball": [None if not np.isfinite(b) else int(b) for b in ball],
    "models": [{"name": name, "colour": COLOURS[k], "state": [str(s) for s in display_state(name)[span]],
                "p": column(models[name][0], "p_dead"), "pStart": column(models[name][0], "p_start"),
                "pStop": column(models[name][0], "p_stop"), "decisions": clip_decisions(name)}
               for k, name in enumerate(chosen)],
    "events": [{"start": int(e.start), "stop": int(e.stop),
                "label": f"{e.kind} {e.minute}", "colour": EVENT_COLOURS.get(e.kind, "#bbbbbb")}
               for e in in_clip.itertuples()],
    "tracking": clip_tracking(clip_start, clip_end),
    "whistles": None if whistles is None else
    [[int(a), int(b)] for a, b in zip(whistles["frame"], whistles["stop"]) if b > clip_start and a < clip_end],
    "spec": None,
}
if show_spec and broadcast_ok and WAV.exists():
    png, fmin, fmax = clip_spectrogram(str(WAV), b0, (clip_end - clip_start) / fps)
    payload["spec"] = {"png": png, "fmin": fmin, "fmax": fmax} if png else None
player = (APP_DIR / "dead_ball_player.html").read_text(encoding="utf-8")
player = (player.replace("__VIDEO_BASE64__", encoded)
                .replace("__PAYLOAD__", json.dumps(payload).replace("<", "\\u003c")))
components.html(player, height=1150 + (90 if payload["spec"] else 0) + (20 if payload["whistles"] is not None else 0),
                scrolling=True)
st.caption(f"Clip: frames {clip_start}–{clip_end - 1} ({timestamp(clip_start, fps)}–"
           f"{timestamp(clip_end, fps)} in the original video).")

with_decisions = [name for name in chosen if explained[name] is not None]
st.subheader("Why the decoder cut here")
if not with_decisions:
    st.info("No decisions.csv for these models (typed decoders, or folders written before "
            "decoding/explain.py: re-run predict or explain).")
else:
    name = st.selectbox("Model", with_decisions,
                        index=with_decisions.index(reference) if reference in with_decisions else 0)
    table = explained[name]
    part = table[(table["stop"] > clip_start) & (table["start"] < clip_end)]
    st.caption("Each row: a kept segment in the clip, and what each part of the score says FOR keeping it "
               "rather than absorbing it into its neighbours. Prior < 0: the duration priors would absorb it. "
               "Margin = prior + TCN + boundary (small = close call). The first and last segment of a half "
               "have no alternative and are not listed.")
    extra = [c for c in ("whistle", "whistle_tcn", "whistle_boundary", "margin_no_whistle") if c in part]
    if extra:
        st.caption("`whistle`: the part of the margin that comes from the whistles (split into `whistle_tcn`, "
                   "through p_dead, and `whistle_boundary`, through the start / stop heads); "
                   "`margin_no_whistle`: the same segment scored without them (< 0: it would not be kept).")
    st.dataframe(part.assign(**{"start time": [timestamp(f, fps) for f in part["start"]]})[
        ["start", "start time", "seconds", "state", "mean_p_dead", "prior", "tcn", "boundary", "margin"] + extra],
        width="stretch", hide_index=True)
