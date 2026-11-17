"""
Streamlit app to review DFL's possession changes one by one and fix them by dragging the bar.
Run it with: .venv\\Scripts\\python.exe -m streamlit run possession_exploration\\app.py
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VIEWER = os.path.join(ROOT, "external", "datasets_visualization")
for path in (VIEWER, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from external import schema  # noqa: E402
from possession_exploration import labelstore as ls  # noqa: E402
import shared  # noqa: E402

# Params
FPS = schema.FPS
LEAD_S = 2.0                      # the replay starts 2 s before the boundary
EVENT_COLUMNS = ["period_id", "timestamp_s", "event_type"]
DEAD_COLUMNS = ["period", "start_s", "end_s", "duration_s", "start_index", "end_index"]
EDITOR_CSS = """
.pe-root{font-family:system-ui;color:#e7eef4;background:#111b27;padding:12px;border-radius:8px}
.pe-controls{display:flex;gap:16px;align-items:center;margin:0 0 8px;font-size:14px}
.pe-controls button{background:#e7eef4;border:0;border-radius:5px;padding:6px 12px;cursor:pointer}
.pe-pitch{width:auto;max-width:100%;max-height:440px;background:#17643e;border-radius:7px;display:block;
          margin:0 auto}
.pe-bar{width:100%;height:84px;margin-top:8px;background:#0b131d;border-radius:5px;outline:none;
        touch-action:none;display:block}
.pe-bar:focus{box-shadow:0 0 0 2px #ffd400}
.pe-help,.pe-legend{font-size:13px;color:#aab6c3;margin-top:6px}
"""

st.set_page_config(page_title="Possession relabelling", page_icon="⚽", layout="wide")


# Loading

def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_resource
def editor():
    with open(os.path.join(HERE, "timeline_editor.js"), encoding="utf-8") as handle:
        return st.components.v2.component("possession_timeline_editor", js=handle.read(), css=EDITOR_CSS)


@st.cache_resource(show_spinner="Loading tracking...")
def load_tracking(source, match, modified):
    # Positions come from the original, only the labels change
    original = schema.load_match(ls.original_path(source, match))
    tracking = shared.tracking_from_match(original, source_format=source)
    return original, tracking


@st.cache_data(show_spinner=False)
def load_labels(path, modified):
    return pd.read_parquet(path, columns=["label"])["label"].to_numpy().astype("<U1")


@st.cache_data(show_spinner="Loading events...")
def load_events(source, match):
    # The ticks on the bar. Empty if there are no event files.
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        if source == "dfl":
            raw = os.path.join(schema.DFL_DIR, "raw")
            names = os.listdir(raw) if os.path.isdir(raw) else []
            info = next((n for n in names if n.startswith("DFL_02_01") and match in n), None)
            events = next((n for n in names if n.startswith("DFL_03_02") and match in n), None)
            if not (info and events):
                return empty
            info, events = os.path.join(raw, info), os.path.join(raw, events)
            table = shared.load_dfl_events(info, events, mtime(info), mtime(events))
        else:
            path = os.path.join(os.path.dirname(ls.original_path(source, match)), match + "_events.csv")
            if not os.path.isfile(path):
                return empty
            table = pd.read_csv(path)
        table = table.dropna(subset=["timestamp_s"])
        return table.assign(event_type=table.event_type.astype(str))[EVENT_COLUMNS]
    except Exception as error:                         # no kloppy or a weird file: just no ticks
        st.sidebar.caption("No event ticks: {}".format(error))
        return empty


@st.cache_data(show_spinner=False)
def git_user():
    try:
        return subprocess.run(["git", "config", "user.name"], capture_output=True, text=True,
                              cwd=ROOT, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def clock(tracking, row):
    t = float(tracking["timestamp_s"][row])
    return "P{} {:02d}:{:05.2f}".format(int(tracking["period_id"][row]), int(t // 60), t % 60)


# Sidebar

records, problems = ls.read_logs()
st.sidebar.title("Possession relabelling")
user = st.sidebar.text_input("Your name (one log per person)", value=git_user(), key="user").strip()
available = {s: ls.matches(s) for s in ls.SOURCES if ls.matches(s)}
if not available:
    st.info("No datasets yet. Run, from the repo root:\n\n"
            "    python possession_exploration/prepare_dfl.py")
    st.stop()
source = st.sidebar.selectbox("Dataset", list(available))
match = st.sidebar.selectbox("Match", available[source])
if problems:
    st.sidebar.warning("{} malformed log lines skipped:\n\n".format(len(problems)) + "\n\n".join(problems[:5]))

working, _ = ls.rebuild(source, match, records)         # so a teammate's edits show up after a pull
original, tracking = load_tracking(source, match, mtime(ls.original_path(source, match)))
label = load_labels(working, mtime(working))
frame, period = original.frame, original.period
reviews = ls.for_match(records, source, match, "review")
table = ls.review_status(ls.transitions(label, period, frame), reviews)

st.sidebar.subheader("Filters")
kinds = st.sidebar.multiselect("Transition type", ls.KINDS, default=list(ls.KINDS))
periods = st.sidebar.multiselect("Period", [1, 2], default=[1, 2])
only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=False)
short = st.sidebar.checkbox("Only next to a short spell", value=False,
                            help="The spell before or after is shorter than the threshold: DFL flicker")
short_s = st.sidebar.slider("Short spell threshold (s)", 0.2, 5.0, 1.0, 0.2, disabled=not short)
st.sidebar.subheader("Label merge (possession model)")
merge_frames = int(st.sidebar.number_input(
    "min_frames", 0, 250, ls.merge_min_frames(), step=1,
    help="The possession model trains on labels where a possession shorter than this (frames, 25 = 1 s) "
         "sandwiched between two of the other team is merged into them (data.min_frames in "
         "possession_model/configs/v1_frozen.toml). Changing it here changes nothing but this view."))
table["merge_removes"] = ls.merge_removes(table, label, period, merge_frames)
only_merged = st.sidebar.checkbox("Only transitions the merge removes", value=False)
st.sidebar.caption("The merge removes {} of {} transitions of this match.".format(
    int(table.merge_removes.sum()), len(table)))
mask = table.kind.isin(kinds) & table.period.isin(periods)
if only_unreviewed:
    mask &= ~table.reviewed
if short:
    mask &= (table.before_s < short_s) | (table.after_s < short_s)
if only_merged:
    mask &= table.merge_removes
view = table[mask].reset_index(drop=True)
width_s = st.sidebar.slider("Replay window around the boundary (s)", 3, 30, 8,
                            help="Changing it drops a drag that is not saved yet")

st.sidebar.subheader("Progress")
st.sidebar.progress(float(table.reviewed.mean()) if len(table) else 0.0,
                    text="{} / {} transitions reviewed in this match".format(int(table.reviewed.sum()), len(table)))
per_user = pd.Series([r[0] for rows in table.reviewed_by for r in rows]).value_counts()
for name, count in per_user.items():
    st.sidebar.caption("{}: {}".format(name, count))
if user:
    st.sidebar.caption("Your log: `possession_exploration/labels/{}.jsonl`. Commit and push it to "
                       "share; a teammate's pull shows up here on the next rerun.".format(ls.safe_user(user)))

# The current transition

st.title("{} · {}".format(source.upper(), match))
st.caption("{} vs {} · labels: {}".format(original.meta.get("home_team"), original.meta.get("away_team"),
                                         original.meta.get("labels", "provider annotation")))
if view.empty:
    st.success("No transition matches the filters.")
    st.stop()

cursor_key = "cursor:{}:{}".format(source, match)
cursor = st.session_state.get(cursor_key, int(view.frame.iloc[0]))
position = int(min(np.searchsorted(view.frame.to_numpy(), cursor), len(view) - 1))


def go(to):
    st.session_state[cursor_key] = int(view.frame.iloc[max(0, min(len(view) - 1, to))])


def next_unreviewed():
    later = view.index[(view.index > position) & ~view.reviewed]
    if len(later):
        go(int(later[0]))


nav = st.columns([1.3, 1.3, 2, 1.2, 4])
nav[0].button("◀ Prev", on_click=go, args=(position - 1,), disabled=position == 0, width="stretch",
              shortcut="Shift+Left")
nav[1].button("Next ▶", on_click=go, args=(position + 1,), disabled=position == len(view) - 1,
              width="stretch", shortcut="Shift+Right")
nav[2].button("Next unreviewed ⏭", on_click=next_unreviewed, width="stretch", shortcut="Shift+Down")
jump = nav[3].number_input("Go to #", 1, len(view), position + 1, label_visibility="collapsed")
if jump - 1 != position:
    go(int(jump) - 1)
    st.rerun()
t = view.iloc[position]
row = int(t["row"])
nav[4].markdown("**Transition {} / {}** · {} · **{} → {}** · {} · spells {:.2f} s | {:.2f} s".format(
    position + 1, len(view), t["kind"], t["from"], t["to"], clock(tracking, row), t["before_s"], t["after_s"]))

if t["reviewed"]:
    st.success("✅ Reviewed by **{}** · {}{}".format(
        t["last_user"], t["last_verdict"], " · " + t["last_note"] if t["last_note"] else ""))
    others = [r for r in t["reviewed_by"] if r[0] != t["last_user"]]
    if others:
        st.caption("Also reviewed by " + ", ".join("{} ({})".format(r[0], r[1]) for r in others))
else:
    st.warning("⚠️ Not reviewed yet")
if t["merge_removes"] and t["last_verdict"] in ls.KEEP_VERDICTS:
    st.info("🔀 The label merge (min_frames = {}) would remove this transition, but it is reviewed "
            "({}): the possession model keeps it.".format(merge_frames, t["last_verdict"]))
elif t["merge_removes"]:
    st.info("🔀 The label merge (min_frames = {}) removes this transition: the short possession next to "
            "it takes the team around it. Is that what the challenge means (a touch without control), or a "
            "real possession? Confirm it (or save an edit) to keep it in the possession model's labels; "
            "leave it unreviewed (or unsure) to let the merge remove it.".format(merge_frames))
actions = st.container(border=True)        # the buttons go here, but we need the drag first

# Replay and the bar you can drag

tid = "{}:{}:{}:{}:{}".format(source, match, int(t["frame"]), t["from"], t["to"])
same = np.flatnonzero(period == period[row])
a = max(int(same[0]), row - width_s * FPS)
b = min(int(same[-1]) + 1, row + width_s * FPS)
selected = np.arange(a, b)
payload, _, _ = shared.player_payload(tracking, selected, load_events(source, match),
                                      pd.DataFrame(columns=DEAD_COLUMNS), initial=max(0, row - a - int(LEAD_S * FPS)))
nonce = st.session_state.setdefault("editor_nonce", 0)
editor_key = "editor:{}:{}:{}:{}".format(tid, a, b, nonce)
data = {key: payload[key] for key in ("t", "period", "fps", "xy", "ball", "player_team", "teams", "events",
                                      "initial")}
data.update(key=editor_key, runs=ls.label_runs(label[a:b]), focus=row - a)
result = editor()(key=editor_key, data=data, on_runs_change=lambda: None, height="content")

current = label[a:b]
proposed = current
dragged = getattr(result, "runs", None)
if dragged:
    try:
        proposed = ls.labels_from_runs(dragged, b - a)
    except ValueError as error:
        st.error("The bar sent an invalid state, ignored: {}".format(error))
edits = ls.diff_edits(current, proposed, frame, a)

# Buttons


def discard():
    st.session_state["editor_nonce"] = st.session_state.get("editor_nonce", 0) + 1


with actions:
    can_save = bool(user)
    if not user:
        st.info("Type your name in the sidebar to save.")
    buttons = st.columns([1.7, 1.7, 1.5, 1.1, 2.5])
    confirm = buttons[0].button("✅ Confirm", disabled=not can_save or bool(edits), width="stretch",
                                help="It is correct as it is: mark it reviewed and go to the next unreviewed",
                                shortcut="Shift+Enter")
    save = buttons[1].button("💾 Save", type="primary", disabled=not can_save or not edits,
                             width="stretch", help="Store the dragged boundaries and mark them reviewed",
                             shortcut="Ctrl+Enter")
    unsure = buttons[2].button("❓ Unsure", disabled=not can_save, width="stretch", shortcut="Shift+U")
    buttons[3].button("↩ Discard", on_click=discard, disabled=not edits, width="stretch",
                      help="Put the bar back as it is saved")
    note = buttons[4].text_input("Note", key="note:" + tid, placeholder="Note (optional)",
                                 label_visibility="collapsed")
    if edits:
        moved = sum(e["stop"] - e["start"] for e in edits)
        st.caption("Dragged: {} frames ({:.2f} s) relabelled. Confirm is off until you 💾 Save or ↩ Discard."
                   .format(moved, moved / FPS))
    else:
        st.caption("Drag the white handles on the bar to move a boundary; Confirm if it is already right.")

base = {"source": source, "match": match}
if confirm or unsure:
    ls.append(user, [dict(base, kind="review", frame=int(t["frame"]), verdict="ok" if confirm else "unsure",
                          note=note, **{"from": t["from"], "to": t["to"]})])
    if confirm:
        next_unreviewed()
    st.rerun()
if save:
    # Transitions the drag created or moved are marked "edited". The one we were looking at is "ok"
    # if the drag didn't touch it. Then jump to the closest transition.
    new_label = ls.apply_edits(label, frame, period, edits)
    before = set(map(tuple, ls.transitions(label, period, frame)[["frame", "from", "to"]].to_numpy().tolist()))
    after = ls.transitions(new_label, period, frame)
    after = after[(after.row >= a) & (after.row < b)]
    new_records = [dict(base, kind="edit", **e) for e in edits]
    for item in after.to_dict("records"):
        key = (item["frame"], item["from"], item["to"])
        if key not in before:
            new_records.append(dict(base, kind="review", frame=int(item["frame"]), verdict="edited", note=note,
                                    **{"from": item["from"], "to": item["to"]}))
        elif key == (int(t["frame"]), t["from"], t["to"]):
            new_records.append(dict(base, kind="review", frame=int(item["frame"]), verdict="ok", note=note,
                                    **{"from": item["from"], "to": item["to"]}))
    ls.append(user, new_records)
    ls.rebuild(source, match, ls.read_logs()[0])
    if len(after):
        st.session_state[cursor_key] = int(after.frame.iloc[int(np.argmin(np.abs(after.row.to_numpy() - row)))])
    discard()
    st.rerun()

with st.expander("All transitions matching the filters ({})".format(len(view))):
    shown = view.assign(time=[clock(tracking, r) for r in view.row], reviewed_by=view.last_user.fillna(""))
    picked = st.dataframe(shown[["time", "kind", "from", "to", "before_s", "after_s", "merge_removes",
                                 "reviewed_by", "last_verdict"]], hide_index=False, on_select="rerun",
                          selection_mode="single-row", key="table:{}:{}".format(source, match))
    chosen = picked.selection.rows[0] if picked.selection.rows else None
    handled_key = "picked:{}:{}".format(source, match)
    if chosen is not None and chosen != st.session_state.get(handled_key):
        st.session_state[handled_key] = chosen       # only react to a new click, not every rerun
        go(chosen)
        st.rerun()
