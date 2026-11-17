"""Review possession models on BAR-ATH: public-window scores, the whole match, and clips around
possession changes (video, 2D pitch, every model's labels and the reference's P(H))."""
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))

import config
import possession_review as lib
from media import video_window
from visualizer import parse_atd

HELP = """
**Where the predictions come from.** One folder per prediction under `POSSESSION_ROOT`
(`cache/possession_model/bar/` by default), written by `python -m possession_model.predict`:
`frames.csv` (frame, `p_h` = P(home has the ball), label H/A/D/X) and `model.json` (run, models,
dead-ball mask, decoder). D comes from the dead-ball model's mask; H/A from the possession model.

**Public window** (frames 11,185-18,684, 17 possessions): the official metrics against the public
ground truth. It is a pipeline check, never a way to choose or tune a model (RULES.md, EVALUATION.md):
one possession moves spellF1 by ~6 points.

**Possession starts** (public window): every GT possession start is matched to the reference's nearest
start of the same team within 1 s: *found* (with its lag: > 0 late, < 0 early), *missed*; and the
reference's starts no GT start explains: *spurious*. *after* says what came before it: the other team
(a change of possession) or D (a restart).

**The whole match** (no ground truth): the timeline shows every prediction over the full video; the
agreement table says how often each one agrees with the reference; *Where the models disagree* lists
the stretches where the predictions shown differ (*team*: some say H, others A; *live/dead*: some say
D), with what each says, to watch them with video.
"""


def configured_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def timestamp(frame, fps):
    s = frame / fps
    return f"{int(s // 60)}:{s % 60:05.2f}"


def modified(path):
    path = Path(path)
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner="Reading predictions...")
def load_prediction(folder, stamp):
    return lib.load_prediction(folder)


@st.cache_data(show_spinner=False)
def load_labels(path, stamp):
    return lib.load_labels(path)


@st.cache_data(show_spinner="Reading pitch tracking...")
def load_tracking(raw_path, raw_modified, metadata_path, metadata_modified, start, end):
    metadata = Path(metadata_path).read_text(encoding="utf-8-sig")
    with Path(raw_path).open(encoding="utf-8-sig") as lines:
        return parse_atd(lines, metadata, start, end)


def clip_tracking(start, end):
    if not (config.ATD_RAW_PATH and config.ATD_METADATA_PATH):
        return None
    try:
        raw, metadata = configured_path(config.ATD_RAW_PATH), configured_path(config.ATD_METADATA_PATH)
        return load_tracking(str(raw), modified(raw), str(metadata), modified(metadata), start, end)
    except (OSError, ValueError, UnicodeError, ET.ParseError) as error:
        st.warning(f"Pitch tracking unavailable: {error}")
        return None


def clip_video(start, end, fps):
    path = configured_path(config.TACTICAL_VIDEO_PATH) if config.TACTICAL_VIDEO_PATH else None
    if path is None or not path.exists():
        return None
    try:
        return video_window(str(path), modified(path), fps, start, end)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        st.warning(f"Could not prepare the video: {error}")
        return None


st.title("Possession review")
with st.expander("How to read this page"):
    st.markdown(HELP)

fps = config.FPS
root = configured_path(getattr(config, "POSSESSION_ROOT", "cache/possession_model/bar"))
found = lib.discover(root)
if not found:
    st.info(f"No predictions in `{root}`. Make one (it reads the match: run it yourself):\n\n"
            "    python -m possession_model.predict --run possession_model/results/<run> "
            "--match data/processed/BAR-ATH.parquet --dead cache/dead_alive/<dead-ball model>/frames.csv "
            "--out cache/possession_model/bar/<name>")
    st.stop()

names = list(found)
default = [n for n in getattr(config, "POSSESSION_DEFAULT", []) if n in found] or names[:3]
chosen = st.sidebar.multiselect("Predictions", names, default=default)
if not chosen:
    st.stop()
reference = st.sidebar.selectbox("Reference (its P(H) curve, its possession starts)", chosen)
with_baselines = st.sidebar.checkbox("Add the baselines", value=False,
                                     help="evaluation/results/baseline_*.csv: H/A from the ball, D from its gaps")

predictions = {n: load_prediction(str(found[n]), modified(found[n] / "frames.csv")) for n in chosen}
labels = {n: predictions[n][0] for n in chosen}
if with_baselines:
    for name, path in config.BASELINE_PATHS.items():
        path = configured_path(path)
        if path.exists():
            labels[f"baseline_{name}"] = load_labels(str(path), modified(path))
for n in chosen:
    st.sidebar.caption(f"**{n}**: {lib.describe(predictions[n][2])}")

gt_path = configured_path(config.GT_PATH) if config.GT_PATH else None
gt = load_labels(str(gt_path), modified(gt_path)) if gt_path is not None and gt_path.exists() else None
w0, w1 = config.WINDOW_START_FRAME, config.WINDOW_END_FRAME

lengths = {n: len(v) for n, v in labels.items()}
if gt is not None:
    lengths["GT"] = len(gt)
if len(set(lengths.values())) > 1:
    st.error(f"The label files do not have the same number of frames: {lengths}")
    st.stop()

# -- scores --------------------------------------------------------------------------------------

st.subheader("Public window")
if gt is None:
    st.info("Set GT_PATH in visualization_app/config.py to score the public window.")
else:
    table = pd.DataFrame({n: lib.window_metrics(v, gt, w0, w1) for n, v in labels.items()}).T
    st.dataframe(table.style.format({"spellF1": "{:.3f}", "macroF1": "{:.3f}", "team acc": "{:.3f}",
                                     "changes F1@1s": "{:.3f}", "lag (s)": "{:.2f}",
                                     "possessions": "{:.0f}", "GT possessions": "{:.0f}"}), width="stretch")
    st.caption(f"Frames {w0}-{w1 - 1} ({timestamp(w0, fps)}-{timestamp(w1, fps)}). A pipeline check: "
               "never choose or tune a model on it.")

st.subheader("Whole match")
whole = pd.DataFrame({n: lib.match_metrics(v, fps) for n, v in labels.items()}).T
st.dataframe(whole.style.format({"possessions": "{:.0f}", "mean s": "{:.1f}", "median s": "{:.1f}",
                                 "< 1 s": "{:.1%}", "H": "{:.1%}", "A": "{:.1%}", "D": "{:.1%}"}), width="stretch")
st.caption("Share of graded frames (not X). No ground truth outside the public window.")

COLOURS = {"H": "#ef4444", "A": "#3b82f6", "D": "#8a93a0"}      # as in possession_player.html
lanes = ([("GT (public window)", gt)] if gt is not None else []) + list(labels.items())
timeline = pd.concat([lib.runs(v).assign(model=n) for n, v in lanes], ignore_index=True)
timeline = timeline[timeline.label.isin(list(COLOURS))]
timeline = timeline.assign(from_min=timeline.start / fps / 60, to_min=timeline.stop / fps / 60,
                           seconds=(timeline.stop - timeline.start) / fps,
                           at=[timestamp(f, fps) for f in timeline.start])
window = pd.DataFrame({"from_min": [w0 / fps / 60], "to_min": [w1 / fps / 60]})
bars = alt.Chart(timeline).mark_rect().encode(
    x=alt.X("from_min:Q", title="minute of the video", scale=alt.Scale(domain=[0, len(labels[reference]) / fps / 60])),
    x2="to_min:Q", y=alt.Y("model:N", sort=[n for n, _ in lanes], title=None),
    color=alt.Color("label:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=["model", "label", "at", alt.Tooltip("seconds:Q", format=".1f"), "start"]).interactive(bind_y=False)
shade = alt.Chart(window).mark_rect(color="#facc15", opacity=0.15).encode(x="from_min:Q", x2="to_min:Q")
st.altair_chart(alt.layer(shade, bars).properties(height=30 * len(lanes) + 60), width="stretch")
st.caption("Whole-match timeline: red H, blue A, grey D, empty X. Scroll to zoom, drag to pan, hover a block "
           "for its time. Yellow: the public window. To watch a moment, use *Where the models disagree* or "
           "*Time* below.")

others = [n for n in labels if n != reference]
if others:
    agree = pd.DataFrame({n: lib.agreement(labels[reference], labels[n]) for n in others}).T
    st.markdown(f"**Agreement with the reference** (`{reference}`), whole match, no ground truth")
    st.dataframe(agree.style.format({"same label": "{:.1%}", "same live/dead": "{:.1%}",
                                     "same team (both live)": "{:.1%}", "frames on the other team": "{:,.0f}"}),
                 width="stretch")

# -- a clip ----------------------------------------------------------------------------------------

st.subheader("Clip")
ref = labels[reference]
sources = ["Frame range", "Where the models disagree (whole match)", "Reference's possession starts", "Time"]
if gt is not None:
    sources = ["Frame range", "Where the models disagree (whole match)", "Possession starts vs GT (public window)",
               "Disagreements with GT (public window)", "Reference's possession starts", "Time"]
source = st.radio("Go to", sources, horizontal=True)
MAX_RANGE_S = 15 * 60       # a longer clip makes the page heavy (every frame's tracking travels)
if source == "Possession starts vs GT (public window)":
    starts = lib.compare_starts(ref, gt, w0, w1)
    st.caption(f"{reference}: {(starts.status == 'found').sum()} found, {(starts.status == 'missed').sum()} missed, "
               f"{(starts.status == 'spurious').sum()} spurious.")
    options = [f"{timestamp(r.frame, fps)} · {r.team} · {r.status}" + (f" ({r.lag_s:+.2f} s)" if r.status == "found" else "")
               + f" · after {r.after}" for r in starts.itertuples()]
    centres = starts.frame.tolist()
elif source == "Disagreements with GT (public window)":
    wrong = lib.disagreements(ref, gt, w0, w1)
    options = [f"{timestamp(r.start, fps)} · {r.seconds:.1f} s · GT {r.gt}, predicted {r.pred}" for r in wrong.itertuples()]
    centres = [(r.start + r.stop) // 2 for r in wrong.itertuples()]
elif source == "Where the models disagree (whole match)":
    k_col, m_col = st.columns(2)
    kind = k_col.radio("Show", ["team (H vs A)", "live/dead", "both"], horizontal=True)
    min_s = m_col.slider("At least (s)", 1, 20, 2)
    split = lib.model_disagreements(labels, min_frames=min_s * fps)
    if kind != "both":
        split = split[split.kind == kind.split()[0]]
    st.caption(f"{len(split)} stretches where the {len(labels)} predictions shown do not agree "
               f"({split.seconds.sum() / 60:.1f} min). Each one: what every prediction says along it.")
    names_short = {n: n.replace("pathcrf_", "").replace("_tcn_bar_boundary", "_tcn").replace("_ens_final3_all7", "_ens")
                   for n in labels}
    options = [f"{timestamp(r['start'], fps)} · {r['seconds']:.1f} s · "
               + " | ".join(f"{names_short[n]} {r[n]}" for n in labels) for r in split.to_dict("records")]
    centres = [(a + b) // 2 for a, b in zip(split.start, split.stop)]
elif source == "Reference's possession starts":
    starts = lib.possession_starts(ref)
    options = [f"{timestamp(r.frame, fps)} · {r.team} · after {r.after} · lasts {r.seconds:.1f} s" for r in starts.itertuples()]
    centres = starts.frame.tolist()
else:
    options, centres = [], []
if source == "Frame range":
    a_col, b_col = st.columns(2)
    first = int(a_col.number_input("From frame", 0, len(ref) - 1, w0, step=fps))
    last = int(b_col.number_input("To frame (excluded)", 1, len(ref), max(first + 1, min(w1, len(ref))), step=fps))
    if last <= first:
        st.warning("The end must come after the start.")
        st.stop()
    if last - first > MAX_RANGE_S * fps:
        st.warning(f"At most {MAX_RANGE_S // 60} min at once: showing {timestamp(first, fps)} + {MAX_RANGE_S // 60} min.")
        last = first + MAX_RANGE_S * fps
    clip_start, clip_end, centre = first, last, first
    st.caption(f"{timestamp(first, fps)}-{timestamp(last, fps)} ({(last - first) / fps:.0f} s). The default is the "
               "public window.")
elif source == "Time":
    minute = st.number_input("Minute of the video (e.g. 63.5)", 0.0, (len(ref) - 1) / fps / 60,
                             round(w0 / fps / 60, 2), step=0.25, format="%.2f")
    centre = min(len(ref) - 1, int(round(minute * 60 * fps)))
    st.caption(f"Frame {centre} ({timestamp(centre, fps)}).")
elif not options:
    st.info("Nothing to list here.")
    st.stop()
else:
    pick = st.selectbox(f"{len(options)} moments", range(len(options)), format_func=lambda k: options[k])
    centre = int(centres[pick])
if source != "Frame range":
    before = st.slider("Seconds before", 1, 30, 6)
    after = st.slider("Seconds after", 1, 30, 6)
    clip_start, clip_end = max(0, centre - before * fps), min(len(ref), centre + after * fps)
show_video = st.checkbox("Video", value=(clip_end - clip_start) <= 90 * fps,
                         help="Off by default for clips over 90 s: encoding a long clip takes a while")

span = slice(clip_start, clip_end)
p_h = predictions[reference][1]
rows = ([{"name": "GT", "labels": "".join(gt[span])}] if gt is not None else []) + \
       [{"name": n, "labels": "".join(v[span])} for n, v in labels.items()]
payload = {"clipStart": clip_start, "clipEnd": clip_end, "fps": fps, "mark": centre, "rows": rows,
           "ph": None if p_h is None else [None if not np.isfinite(q) or ref[clip_start + k] not in lib.TEAMS
                                           else round(float(q), 3) for k, q in enumerate(p_h[span])],
           "phName": reference, "tracking": clip_tracking(clip_start, clip_end),
           "video": clip_video(clip_start, clip_end, fps) if show_video else None}
player = (APP_DIR / "possession_player.html").read_text(encoding="utf-8")
player = player.replace("__PAYLOAD__", json.dumps(payload).replace("<", "\\u003c"))
st.iframe(player, height=(820 if payload["video"] else 640) + 28 * len(rows))
st.caption(f"Clip: frames {clip_start}-{clip_end - 1} ({timestamp(clip_start, fps)}-{timestamp(clip_end, fps)}).")
