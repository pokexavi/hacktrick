import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import streamlit as st
from config import COMPETITIONS, DFL_DIR, SYNTHETIC_DIR
from open_datasets import (load_metrica_csv_events, load_metrica_csv_tracking,
                           load_metrica_epts_tracking, load_metrica_json_events, metrica_possession,
                           open_sources, POSSESSION_AWAY, POSSESSION_DEAD, POSSESSION_HOME,
                           POSSESSION_UNKNOWN)


def read_noise_profile(content):
    """Identify combined v1, player-only v2, or ball-only v2 aggregate JSON."""
    profile = json.loads(content)
    if not isinstance(profile, dict):
        raise ValueError("The JSON root must be an object")
    if profile.get("kind") == "ball_v2" and profile.get("version") == 1:
        q = profile.get("gap_frames_q")
        if not isinstance(q, list) or len(q) != 41:
            raise ValueError("Ball profile needs 41 gap-length quantiles")
        fraction = profile.get("target_lost_fraction")
        if not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            raise ValueError("Ball profile needs a target_lost_fraction between 0 and 1")
        return "ball_v2", profile
    home = profile.get("home")
    away = profile.get("away")
    if not isinstance(home, dict) or not isinstance(away, dict):
        raise ValueError("The profile needs home and away team objects")
    if profile.get("version") == 1 and "unique_players_mean" in home:
        required = ("n_slots", "unique_players_mean", "unlinked_end_context_rate",
                    "cross_slot_link_frac", "cross_slot_links_per_player_frame")
        missing = [f"{side}.{key}" for side in ("home", "away") for key in required
                   if profile[side].get(key) is None]
        if missing:
            raise ValueError("Missing player-profile fields: " + ", ".join(missing))
        return "players_v2", profile
    required = ("ball_gap_q", "ball_lost_frac", "roughness_m")
    missing = [key for key in required if profile.get(key) is None]
    missing.extend(f"{side}.{key}" for side in ("home", "away")
                   for key in ("n_slots", "tracked_mean", "tracklet_q")
                   if profile[side].get(key) is None)
    if missing:
        raise ValueError("Missing combined-profile fields: " + ", ".join(missing))
    if not 0 < profile["ball_lost_frac"] < 1:
        raise ValueError("ball_lost_frac must be between 0 and 1")
    return "combined_v1", profile


def sources():
    """Only the public open datasets and the committed synthetic sample are discoverable."""
    available = {}
    for match in COMPETITIONS:
        stem = f"DFL-MAT-{match}"
        parquet = DFL_DIR / "processed" / f"{stem}.parquet"
        sidecar = parquet.with_suffix(".json")
        if parquet.is_file() and sidecar.is_file():
            raw = DFL_DIR / "raw"
            suffix = f"_DFL-MAT-{match}.xml"
            competition = COMPETITIONS[match]
            available[f"DFL · {match}"] = {
                "kind": "dfl", "dataset": "DFL", "match": match, "tracking": parquet,
                "meta": raw / f"DFL_02_01_matchinformation_{competition}{suffix}",
                "events": raw / f"DFL_03_02_events_raw_{competition}{suffix}",
            }
    available.update(open_sources())
    for name in ("tactical", "broadcast"):
        stem = f"BAR-ATH_{name}"
        folder = SYNTHETIC_DIR / name
        meta = folder / f"{stem}_atd.xml"
        raw = folder / f"{stem}_atd.txt"
        if meta.is_file() and raw.is_file():
            available[f"Synthetic · {name.title()}"] = {
                "kind": "synthetic", "dataset": "Synthetic", "match": stem, "tracking": raw,
                "meta": meta, "events": folder / f"{stem}_smart_tagging.xml",
            }
    return available


def choose_source(available, key):
    """Sidebar pickers, dataset then match: (the source's name, the source)."""
    datasets = list(dict.fromkeys(source["dataset"] for source in available.values()))
    dataset = st.sidebar.selectbox("Dataset", datasets, key=f"{key}_dataset")
    names = [name for name, source in available.items() if source["dataset"] == dataset]
    name = st.sidebar.selectbox("Match", names, key=f"{key}_match_{dataset}")
    return name, available[name]


def has_events(source):
    return source["events"].is_file() and (source["meta"] is None or source["meta"].is_file())


def _stamp(path):
    return path.stat().st_mtime_ns if path.is_file() else 0


def active_corruption():
    """The corruptions applied on the Corruptions page, or None: they hold for every page."""
    return st.session_state.get("corruption")


def describe_corruption(choice):
    text = f"players {choice['players']} · ball {choice['ball']} · dead ball {choice['dead_ball']}"
    if choice.get("swaps", "none") != "none":
        text = f"swaps {choice['swaps']} · " + text
    return text + f" · copy {choice['copy']}"


def corruption_banner():
    """Tell, in the sidebar of a page, whether its DFL matches are corrupted."""
    if active_corruption():
        st.sidebar.warning("**Corrupted DFL:** " + describe_corruption(active_corruption()))
    else:
        st.sidebar.caption("Clean DFL. Apply corruptions on the Corruptions page.")


def tracking_for(source):
    """The tracking a page shows: a DFL match carries the active corruptions."""
    choice = active_corruption()
    if choice and source["kind"] == "dfl":
        from recipes import corrupted_tracking
        return corrupted_tracking(source, choice)[0]
    return clean_tracking_for(source)


def clean_tracking_for(source):
    path = source["tracking"]
    meta = source["meta"]
    kind = source["kind"]
    if kind == "dfl":
        return load_dfl_tracking(str(path), _stamp(path), _stamp(path.with_suffix(".json")))
    if kind == "metrica_csv":
        away = source["away"]
        return load_metrica_csv_tracking(str(path), str(away), _stamp(path), _stamp(away))
    if kind == "metrica_epts":
        return load_metrica_epts_tracking(str(meta), str(path), _stamp(meta), _stamp(path))
    return load_synthetic_tracking(str(meta), str(path), _stamp(meta), _stamp(path))


def _period_starts(source):
    """{period: its first tracking frame}: the clock of the open datasets' events."""
    tracking = clean_tracking_for(source)
    frames, periods = tracking["frame_id"], tracking["period_id"]
    return {int(period): int(frames[np.argmax(periods == period)]) for period in np.unique(periods)}


def events_for(source):
    event = source["events"]
    if not has_events(source):
        return pd.DataFrame()
    kind = source["kind"]
    if kind == "synthetic":
        return load_synthetic_events(str(event), _stamp(event))
    meta = source["meta"]
    if kind == "dfl":
        return load_dfl_events(str(meta), str(event), _stamp(meta), _stamp(event))
    starts = _period_starts(source)
    if kind == "metrica_csv":
        return load_metrica_csv_events(str(event), starts, _stamp(event))
    return load_metrica_json_events(str(event), starts, _stamp(event))


@st.cache_resource(show_spinner="Loading DFL tracking...")
def load_dfl_tracking(path, modified, sidecar_modified):
    from external.schema import load_match
    return tracking_from_match(load_match(path))


def tracking_from_match(match, source_format="DFL Parquet"):
    """Adapt a clean or degraded schema.Match to the shared 2D player format."""
    length, width = match.pitch
    ball = match.ball[:, :2].copy()
    ball[:, 0] = ball[:, 0] / length + 0.5
    ball[:, 1] = ball[:, 1] / width + 0.5
    players = np.concatenate((match.home, match.away), axis=1)
    players = players.copy()
    players[:, :, 0] = players[:, :, 0] / length + 0.5
    players[:, :, 1] = players[:, :, 1] / width + 0.5
    times = np.full(len(match.frame), np.nan, dtype=np.float32)
    for period in (1, 2):
        ix = np.flatnonzero(match.period == period)
        if len(ix):
            times[ix] = (match.frame[ix] - match.frame[ix[0]]) / 25.0
    labels = np.where(match.label == "D", 1,
                      np.where(np.isin(match.label, ("H", "A")), 0, -1)).astype(np.int8)
    possession = np.select([match.label == "H", match.label == "A", match.label == "D"],
                           [POSSESSION_HOME, POSSESSION_AWAY, POSSESSION_DEAD],
                           POSSESSION_UNKNOWN).astype(np.int8)
    return {"possession": possession,
"frame_id": match.frame, "period_id": match.period,
            "timestamp_s": times, "ball_xy": ball, "ball_z": match.ball[:, 2],
            "players_xy": players,
            "player_team": np.r_[np.zeros(match.home.shape[1], dtype=np.int8),
                                 np.ones(match.away.shape[1], dtype=np.int8)],
            "team_names": np.array([match.meta["home_team"], match.meta["away_team"]]),
            "label": labels, "fps": 25, "source_format": source_format,
            "label_note": "Labels are the Sportec ball-state annotation."}


@st.cache_resource(show_spinner="Loading synthetic tracking...")
def load_synthetic_tracking(meta_path, raw_path, meta_modified, raw_modified):
    from kloppy import metrica
    dataset = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path,
                                         coordinates="kloppy")
    df = dataset.to_df()
    teams = list(dataset.metadata.teams)
    slots = [str(player.player_id) for team in teams for player in team.players]
    xy = np.full((len(df), len(slots), 2), np.nan, dtype=np.float32)
    for index, slot in enumerate(slots):
        for axis, column in enumerate((f"{slot}_x", f"{slot}_y")):
            if column in df:
                xy[:, index, axis] = pd.to_numeric(df[column], errors="coerce").to_numpy(np.float32)
    ball = df[["ball_x", "ball_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    ball_z = (pd.to_numeric(df["ball_z"], errors="coerce").to_numpy(np.float32)
              if "ball_z" in df else np.full(len(df), np.nan, np.float32))
    frame = pd.to_numeric(df["frame_id"], errors="raise").to_numpy(np.int64)
    return {"frame_id": frame, "period_id": np.ones(len(df), dtype=np.int8),
            "timestamp_s": ((frame - frame[0]) / 25).astype(np.float32),
            "ball_xy": ball, "ball_z": ball_z, "players_xy": xy,
            "player_team": np.r_[np.zeros(len(teams[0].players), np.int8),
                                 np.ones(len(teams[1].players), np.int8)],
            "team_names": np.array([team.name for team in teams]),
            "label": np.full(len(df), -1, np.int8),
            "possession": np.full(len(df), POSSESSION_UNKNOWN, np.int8), "fps": 25,
            "source_format": "Synthetic ATD (no dead-ball labels)",
            "label_note": "This synthetic sample has no dead-ball labels."}


@st.cache_data(show_spinner="Loading DFL events...")
def load_dfl_events(meta_path, event_path, meta_modified, event_modified):
    from kloppy import sportec
    events = sportec.load_event(meta_data=meta_path, event_data=event_path,
                                coordinates="kloppy").to_df()
    columns = [column for column in ("event_id", "event_type", "period_id", "timestamp",
               "end_timestamp", "team_id", "player_id", "set_piece_type", "result",
               "success", "coordinates_x", "coordinates_y") if column in events]
    events = events[columns].copy()
    for column in ("timestamp", "end_timestamp"):
        if column in events:
            events[column] = pd.to_timedelta(events[column]).dt.total_seconds()
    events = events.rename(columns={"timestamp": "timestamp_s",
                                    "end_timestamp": "end_timestamp_s"})
    for column in ("event_id", "set_piece_type", "result"):
        if column not in events:
            events[column] = pd.NA
    events["event_id"] = events["event_id"].astype(str)
    return events


@st.cache_data(show_spinner="Loading synthetic tags...")
def load_synthetic_events(path, modified):
    root = ET.parse(path).getroot()
    rows = []
    for instance in root.findall(".//ALL_INSTANCES/instance"):
        labels = {label.findtext("group"): label.findtext("text")
                  for label in instance.findall("label")}
        code = instance.findtext("code") or "EVENT"
        detail = labels.get("Type")
        rows.append({"event_id": instance.findtext("ID"), "event_type":
                     f"{code}: {detail}" if detail else code,
                     "period_id": 1, "timestamp_s": float(instance.findtext("start")),
                     "end_timestamp_s": float(instance.findtext("end")),
                     "team_id": labels.get("Team"), "player_id": None,
                     "set_piece_type": detail if code == "SET PIECES" else None,
                     "result": None})
    return pd.DataFrame(rows)


def dead_intervals(tracking):
    return intervals(tracking["label"], tracking["period_id"], tracking["timestamp_s"],
                     1 / tracking["fps"])


def intervals(labels, periods, times, frame_s=0.04):
    boundaries = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) |
                                      (periods[1:] != periods[:-1])) + 1, len(labels)]
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if labels[start] == 1:
            rows.append({"period": int(periods[start]), "start_s": float(times[start]),
                         "end_s": float(times[end - 1]) + frame_s,
                         "duration_s": float(times[end - 1] - times[start]) + frame_s,
                         "start_index": int(start), "end_index": int(end)})
    return pd.DataFrame(rows, columns=["period", "start_s", "end_s", "duration_s",
                                       "start_index", "end_index"])


def render_player(data):
    payload = (json.dumps(data, allow_nan=False).replace("<", r"\u003c")
               .replace(">", r"\u003e").replace("&", r"\u0026"))
    html = """
<style>
body{font-family:system-ui;margin:0;color:#e7eef4;background:#111b27} .wrap{padding:12px}
.controls{display:flex;gap:12px;align-items:center;margin:0 0 10px}.controls button{background:#e7eef4;border:0;border-radius:5px;padding:7px 12px;cursor:pointer}
input[type=range]{flex:1} #pitch{width:100%;max-height:650px;background:#17643e;border-radius:7px}
.meta{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0;font-size:14px}.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:4px}
#timeline{width:100%;height:54px;cursor:pointer;background:#253246;border-radius:5px}
</style><div class="wrap"><div class="controls"><button id="play">Play</button><input id="seek" type="range" min="0" value="0"><span id="clock"></span></div>
<canvas id="pitch" width="1050" height="680"></canvas><canvas id="timeline" width="1050" height="54"></canvas>
<div class="meta"><span><i class="swatch" style="background:#3f8fe0"></i>H: <span id="home"></span></span><span><i class="swatch" style="background:#d94a4a"></i>A: <span id="away"></span></span><span><i class="swatch" style="background:#8a93a0"></i>Dead ball</span><span><i class="swatch" style="background:#253246;box-shadow:inset 0 0 0 1px #8a93a0"></i>No possession data</span><span><i class="swatch" style="background:white"></i>Ball</span><span id="height"></span><span id="state"></span><span id="event"></span></div></div>
<script>
const D=PAYLOAD; const pitch=document.getElementById('pitch'), ctx=pitch.getContext('2d');
const timeline=document.getElementById('timeline'), tx=timeline.getContext('2d');
const seek=document.getElementById('seek');seek.max=D.t.length-1;
document.getElementById('home').textContent=D.teams[0];document.getElementById('away').textContent=D.teams[1];
let frame=Math.max(0,Math.min(D.t.length-1,D.initial||0)),timer=null; const fps=D.fps;
function line(x1,y1,x2,y2){ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke()}
function circle(x,y,r,fill){ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);if(fill)ctx.fill();else ctx.stroke()}
function draw(){
 const w=pitch.width,h=pitch.height,m=24,px=x=>m+x*(w-2*m),py=y=>m+y*(h-2*m);
 ctx.fillStyle='#17643e';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(255,255,255,.8)';ctx.lineWidth=2;
 ctx.strokeRect(m,m,w-2*m,h-2*m);line(w/2,m,w/2,h-m);circle(w/2,h/2,65,false);
 ctx.strokeRect(m,m+145,160,h-2*m-290);ctx.strokeRect(w-m-160,m+145,160,h-2*m-290);
 ctx.strokeRect(m,m+225,55,h-2*m-450);ctx.strokeRect(w-m-55,m+225,55,h-2*m-450);
 for(let p=0;p<D.xy[frame].length;p++){const pos=D.xy[frame][p];if(pos[0]===null||pos[1]===null)continue;
   ctx.fillStyle=D.player_team[p]===0?'#3f8fe0':'#d94a4a';circle(px(pos[0]),py(pos[1]),9,true);
   ctx.fillStyle='#07131d';ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(String(p+1),px(pos[0]),py(pos[1])+3);
   if((D.highlight||[]).includes(p)){ctx.strokeStyle='#ffd400';ctx.lineWidth=4;circle(px(pos[0]),py(pos[1]),15,false);ctx.lineWidth=2;ctx.strokeStyle='rgba(255,255,255,.8)'}
 }
 const ball=D.ball[frame];if(ball[0]!==null&&ball[1]!==null){ctx.fillStyle='white';circle(px(ball[0]),py(ball[1]),5,true);ctx.strokeStyle='#111';ctx.stroke()}
 seek.value=frame;document.getElementById('clock').textContent=`P${D.period} ${Math.floor(D.t[frame]/60)}:${String(Math.floor(D.t[frame]%60)).padStart(2,'0')}.${String(Math.floor(D.t[frame]*100)%100).padStart(2,'0')}`;
 document.getElementById('height').textContent=D.ball_z[frame]===null?'Ball height: unavailable':`Ball height: ${D.ball_z[frame].toFixed(2)}`;
 document.getElementById('state').textContent=D.possession[frame]===2?'Now: dead ball':D.possession[frame]===0?'Now: H':D.possession[frame]===1?'Now: A':'Now: no possession data';
 const nearby=D.events.filter(e=>Math.abs(e.t-D.t[frame])<0.6);document.getElementById('event').textContent=nearby.length?'Event: '+nearby.map(e=>e.type).join(', '):'';
 tx.clearRect(0,0,timeline.width,timeline.height);
 const colors={0:'#3f8fe0',1:'#d94a4a',2:'#8a93a0','-1':'#253246'};
 for(const run of D.possession_runs){tx.fillStyle=colors[run[2]];tx.fillRect(run[0]/D.t.length*timeline.width,0,Math.max(1,(run[1]-run[0])/D.t.length*timeline.width),54)}
 for(const e of D.events){const i=Math.max(0,Math.min(D.t.length-1,Math.round((e.t-D.t[0])*fps)));tx.fillStyle='#f5d87a';tx.fillRect(i/D.t.length*timeline.width,0,2,18)}
 tx.fillStyle='white';tx.fillRect(frame/D.t.length*timeline.width,0,2,54);
}
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}else{timer=setInterval(()=>{frame=(frame+1)%D.t.length;draw()},1000/fps);document.getElementById('play').textContent='Pause'}};
seek.oninput=()=>{frame=Number(seek.value);draw()};timeline.onclick=e=>{frame=Math.max(0,Math.min(D.t.length-1,Math.floor(e.offsetX/timeline.clientWidth*D.t.length)));draw()};draw();
</script>""".replace("PAYLOAD", payload)
    st.iframe(html, height=840)


def clean(array):
    return json.loads(json.dumps(array.tolist(), allow_nan=True).replace("NaN", "null"))


def possession_for(tracking, events):
    """Per-frame POSSESSION_* codes. Metrica has none of its own: they come from its events."""
    if "possession" in tracking:
        return tracking["possession"]
    return metrica_possession(tracking, events)


def _runs(codes):
    """[[start, stop, code]] of the maximal runs of one code."""
    edges = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
    return [[int(start), int(stop), int(codes[start])] for start, stop in zip(edges[:-1], edges[1:])]


def player_payload(tracking, selected, events, dead, initial=0):
    periods = tracking["period_id"]
    times = tracking["timestamp_s"]
    possession = possession_for(tracking, events)[selected]
    period = int(periods[selected[0]])
    event_window = events[(events.period_id == period) &
                          (events.timestamp_s >= times[selected[0]]) &
                          (events.timestamp_s <= times[selected[-1]])] if not events.empty else events
    window_dead = dead[(dead.period == period) & (dead.end_index > selected[0]) &
                       (dead.start_index <= selected[-1])] if not dead.empty else dead
    payload = {
        "t": clean(times[selected]), "period": period, "fps": tracking["fps"],
        "xy": clean(tracking["players_xy"][selected]),
        "ball": clean(tracking["ball_xy"][selected]),
        "ball_z": clean(tracking["ball_z"][selected]),
        "possession": possession.tolist(), "possession_runs": _runs(possession),
        "player_team": tracking["player_team"].tolist(),
        "teams": tracking["team_names"].tolist(),
        "events": [{"t": float(row.timestamp_s), "type": str(row.event_type)}
                   for row in event_window.itertuples()] if not event_window.empty else [],
        "initial": int(initial),
    }
    return payload, event_window, window_dead


