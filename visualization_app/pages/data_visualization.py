"""View the public GT window with aligned predictions and automatic tagging."""
import base64
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
import config
from media import video_window
from ball_visibility import parse_visibility
from visualizer import parse_labels, parse_events, segments, payload_json, parse_atd
from possession_review import discover


def configured_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@st.cache_data(show_spinner="Reading ball availability...")
def ball_availability(path, modified):
    with Path(path).open(encoding="utf-8-sig") as lines:
        return parse_visibility(lines)




st.title("Data visualization")
st.caption("Tactical video | Ground truth | Baselines | Smart Tagging")
st.caption("Configure automatic inputs in visualization_app/config.py. "
           "Smart Tagging is automatic annotation, not ground truth.")
with st.sidebar:
    st.header("Sources and alignment")
    fps = st.number_input("Annotation FPS", min_value=1, max_value=120, value=config.FPS)
    start = st.number_input("Window start frame (inclusive)", min_value=0, value=config.WINDOW_START_FRAME)
    end = st.number_input("Window end frame (exclusive)", min_value=1, value=config.WINDOW_END_FRAME)
    st.caption("Ground truth: fixed public five-minute annotation.")
    show_baselines = st.checkbox("Load baselines", value=True)
    possession_models = discover(configured_path(getattr(config, "POSSESSION_ROOT", "cache/possession_model/bar")))
    chosen_models = st.multiselect(
        "Possession models", list(possession_models),
        default=[n for n in getattr(config, "POSSESSION_DEFAULT", []) if n in possession_models],
        help="Every folder under POSSESSION_ROOT (the Possession review page's predictions)")
    predictions = st.file_uploader("Full-match predictions (frame,label CSV)", type=["csv"],
                                   accept_multiple_files=True,
                                   help="Use original zero-based tactical video frames. The GT window is selected automatically.")
    show_atd = st.checkbox("Show ATD pitch view", value=False,
                           help="Loads ATD_METADATA_PATH and ATD_RAW_PATH from config.py.")
    offset = st.number_input("Original first frame of manually selected video", min_value=0, value=0,
                            help="Use 0 for the full tactical video; use the clip start frame for a cropped video.")
    st.caption("H: home | A: away | D: dead ball | X: unavailable GT / prediction abstention.")
if end <= start:
    st.error("Window end must be greater than window start.")
    st.stop()
try:
    gt = parse_labels(configured_path(config.GT_PATH).read_text(encoding="utf-8-sig"))
    rows = [{"name": "GT", "segments": segments(gt)}]
    paths = dict(config.BASELINE_PATHS) if show_baselines else {}
    paths.update(config.PREDICTION_PATHS)
    paths.update({name: possession_models[name] / "labels.csv" for name in chosen_models
                  if (possession_models[name] / "labels.csv").exists()})
    for name, value in paths.items():
        rows.append({"name": name, "segments": segments(parse_labels(
            configured_path(value).read_text(encoding="utf-8-sig")))})
    for upload in predictions:
        rows.append({"name": upload.name, "segments": segments(parse_labels(upload.getvalue().decode("utf-8-sig")))})
    event_text = (configured_path(config.SMART_TAGGING_PATH).read_text(encoding="utf-8-sig")
                  if config.SMART_TAGGING_PATH else None)
    events = parse_events(event_text, fps) if event_text else []
    # Full-match labels and tags retain original frame indices; only filter the visible window.
    for row in rows:
        row["segments"] = [[max(a, start), min(b, end), label]
                           for a, b, label in row["segments"] if b > start and a < end]
    events = [event for event in events if event["end"] > start and event["start"] < end]
    ball_missing = None
    if config.ATD_RAW_PATH:
        raw_path = configured_path(config.ATD_RAW_PATH)
        visible = ball_availability(str(raw_path), raw_path.stat().st_mtime_ns)
        ball_missing = {"name": "ball_data_missing", "kind": "availability",
                        "segments": segments({frame: "available" if visible[frame] else "missing"
                                               for frame in range(start, min(end, len(visible)))})}
    else:
        st.info("Set ATD_RAW_PATH in config.py to display the ball_data_missing timeline.")
    tracking = None
    if show_atd:
        if config.ATD_METADATA_PATH and config.ATD_RAW_PATH:
            metadata = configured_path(config.ATD_METADATA_PATH).read_text(encoding="utf-8-sig")
            with configured_path(config.ATD_RAW_PATH).open(encoding="utf-8-sig") as lines:
                tracking = parse_atd(lines, metadata, start, end)
        else:
            st.info("Set ATD_METADATA_PATH and ATD_RAW_PATH in visualization_app/config.py to show tracking.")
    video_src = ""
    if config.TACTICAL_VIDEO_PATH:
        path = configured_path(config.TACTICAL_VIDEO_PATH)
        video_src = "data:video/mp4;base64," + video_window(str(path), path.stat().st_mtime_ns, fps, start, end)
        offset = start
except (ValueError, OSError, UnicodeError, ET.ParseError, subprocess.CalledProcessError) as error:
    st.error("Could not load configured inputs: " + str(error))
    st.stop()
if not event_text:
    st.info("Set SMART_TAGGING_PATH in visualization_app/config.py to show tagging events.")
selected = st.multiselect("Visible predictions", [row["name"] for row in rows[1:]],
                         default=[row["name"] for row in rows[1:]])
rows = rows[:1] + [row for row in rows[1:] if row["name"] in selected]
payload = payload_json(dict(rows=rows, events=events, fps=fps, offset=offset,
                           start=start, end=end, videoSrc=video_src, tracking=tracking, ballMissing=ball_missing))
template = (APP_DIR / "visualizer.html").read_text(encoding="utf-8")
components.html(template.replace("__PAYLOAD__", payload), height=1100, scrolling=True)
