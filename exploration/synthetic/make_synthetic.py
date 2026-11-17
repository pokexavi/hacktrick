"""Generate a SYNTHETIC match bundle: exactly the file formats of the real one, made-up values.

Nothing here is read from data/. The formats were taken from DATA.md, the starter loaders, kloppy's
EPTS parser, and exploration/format_probe.py (which prints the real files' structure -- tag paths,
key names, field counts -- and never their values). Every number written below is invented: the
match is a random simulation of possessions, stoppages, 22 players and a ball.

Two sizes:

    python exploration/synthetic/make_synthetic.py
        30 s sample, written to exploration/synthetic/{tactical,broadcast}/ and committed. For
        reading the formats with your own eyes (00_data_formats.ipynb).

    python exploration/synthetic/make_synthetic.py --full --out cache/synthetic_full [--video]
        Full length -- 199,319 tactical frames, the real graded halves, 164,488 broadcast frames --
        so every exploration notebook can be run end to end on fake data:
            $env:MTAH_DATA_DIR = "cache/synthetic_full"
        Also writes BAR-ATH_synthetic_labels.csv (the simulation's own H/A/D/X per frame) and,
        with --video, a tactical mp4 of the first --video-until frames (enough to cover the
        public window; the real video is 199,319 frames, this one stops early).

The layout mirrors data/ so games.asset_paths(data_dir=...) finds everything.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, SOURCES, asset_paths, half_video_blocks  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]                 # metres, only for speeds
N_SLOTS = {"tactical": 46, "broadcast": 51}           # as in the real bundle
TEAM_IDS = ("SYN_HOME", "SYN_AWAY")
TEAM_NAMES = ("Barcelona", "Athletic Club")
ST_TEAM = ("Barcelona", "Atletic Club")               # Smart Tagging spelling, per DATA.md
SECOND_HALF_KICKOFF = 113742                          # DATA.md / FAQ.md
SMALL_FRAMES, SMALL_KICKOFF, SMALL_BROADCAST = 750, 100, 600
H, A, D, OUT = 0, 1, 2, -1

# 4-3-3 in normalised coordinates for a team attacking towards x = 1. Invented, not measured.
FORMATION = np.array([[0.05, 0.50], [0.22, 0.15], [0.20, 0.38], [0.20, 0.62], [0.22, 0.85],
                      [0.40, 0.30], [0.36, 0.50], [0.40, 0.70], [0.58, 0.18], [0.62, 0.50],
                      [0.58, 0.82]])
N_PLAYERS = 22

# Smart Tagging vocabulary, from DATA.md.
ATTACK_CODES = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                "LONG BALL", "ATTACKING TRANSITION", "BALL IN THE BOX", "PLAYERS IN THE BOX")
DEFEND_CODES = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
ALL_CODES = ATTACK_CODES + DEFEND_CODES + ("SET PIECES", "GOALS")
SET_PIECE_TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick")


# -- the simulated match -----------------------------------------------------------------------

def timeline(full):
    """(n_video_frames, [(first, last) per half], [kickoff per half]) in video frames."""
    if full:
        halves = [(first, last) for _, first, last in half_video_blocks(GAME)]
        return SOURCES["tactical"]["n_video_frames"], halves, [halves[0][0], SECOND_HALF_KICKOFF]
    return SMALL_FRAMES, [(25, SMALL_FRAMES - 1)], [SMALL_KICKOFF]


def simulate_labels(n, halves, kickoffs, rng):
    """Per-frame H/A/D inside the halves, OUT elsewhere. Possessions ~12 s, some stoppages."""
    labels = np.full(n, OUT, dtype=np.int8)
    for index, ((first, last), kickoff) in enumerate(zip(halves, kickoffs)):
        labels[first:kickoff] = D
        t, team = kickoff, (H if index == 0 else A)
        while t <= last:
            duration = int(rng.exponential(10 * FPS)) + 2 * FPS
            labels[t:min(t + duration, last + 1)] = team
            t += duration
            if rng.random() < 0.35:
                stop = int(rng.uniform(4, 35) * FPS)
                labels[t:min(t + stop, last + 1)] = D
                t += stop
                team = int(rng.integers(2))
            else:
                team = 1 - team
    return labels


def runs(values):
    """[(start, stop, value)] constant runs."""
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(a), int(b), int(values[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def smooth_noise(rng, shape, a_vel=0.97, a_pos=0.995):
    """Smooth wander: AR(1) velocity integrated by a leaky AR(1). Unit-ish std."""
    eps = rng.standard_normal(shape)
    out = lfilter([1.0], [1.0, -a_pos], lfilter([1.0], [1.0, -a_vel], eps, axis=0), axis=0)
    return out / (out.std() + 1e-12)


def attack_sign(n, halves):
    """+1 where the home side attacks towards x = 1 (first half), -1 in the second."""
    sign = np.ones(n)
    if len(halves) > 1:
        sign[halves[1][0] - (halves[1][0] - halves[0][1]) // 2:] = -1
    return sign


def simulate_positions(labels, halves, rng):
    """(n, 22, 2) normalised positions of everyone, tracked or not. Players 0-10 home."""
    n = len(labels)
    sign = attack_sign(n, halves)
    possessor = np.where(labels == H, 1.0, np.where(labels == A, -1.0, 0.0))
    target_x = 0.5 + 0.18 * possessor * sign
    focus_x = lfilter([0.002], [1.0, -0.998], target_x - 0.5) + 0.5 \
        + 0.12 * smooth_noise(rng, n, 0.99, 0.999)
    focus_y = 0.5 + 0.15 * smooth_noise(rng, n, 0.99, 0.999)

    # Stoppages: everyone slows down -- the velocity driving the wander is scaled down.
    calm = np.where(labels == D, 0.3, 1.0)[:, None, None]
    eps = rng.standard_normal((n, N_PLAYERS, 2)) * calm
    wander = lfilter([1.0], [1.0, -0.995], lfilter([1.0], [1.0, -0.97], eps, axis=0), axis=0)
    wander *= 0.035 / (wander.std() + 1e-12)

    positions = np.empty((n, N_PLAYERS, 2))
    for player in range(N_PLAYERS):
        team_sign = sign if player < 11 else -sign
        anchor = FORMATION[player % 11]
        positions[:, player, 0] = focus_x + team_sign * (anchor[0] - 0.45) * 0.6
        positions[:, player, 1] = 0.5 + (anchor[1] - 0.5) * 0.8 + 0.3 * (focus_y - 0.5)
    positions += wander
    return np.clip(positions, 0.005, 0.995)


def simulate_ball(labels, positions, rng):
    """(n, 2) true ball position: carried, passed between team-mates, still during stoppages."""
    n = len(labels)
    ball = np.full((n, 2), np.nan)
    last = np.array([0.5, 0.5])
    for start, stop, label in runs(labels):
        if label == OUT:
            continue
        if label == D:
            # Out of play: the ball sits just over the nearest line until the restart.
            spot = last.copy()
            if min(spot[1], 1 - spot[1]) < min(spot[0], 1 - spot[0]):
                spot[1] = -0.01 if spot[1] < 0.5 else 1.01
            else:
                spot[0] = -0.01 if spot[0] < 0.5 else 1.01
            ball[start:stop] = spot
            continue
        team_players = np.arange(11) + (0 if label == H else 11)
        carrier = int(rng.choice(team_players[1:]))
        t = start
        while t < stop:
            dwell = int(rng.uniform(20, 60))
            ball[t:min(t + dwell, stop)] = positions[t:min(t + dwell, stop), carrier] + 0.004
            t += dwell
            if t >= stop:
                break
            receiver = int(rng.choice(team_players[team_players != carrier]))
            flight = int(rng.uniform(8, 25))
            end = min(t + flight, stop)
            alpha = np.linspace(0, 1, end - t, endpoint=False)[:, None]
            ball[t:end] = (1 - alpha) * positions[t:end, carrier] + alpha * positions[t:end, receiver]
            t, carrier = end, receiver
        last = ball[stop - 1]
    return ball


def visibility_mask(labels, rng, visible_mean=55, missing_mean=38, dead_missing_mean=70):
    """True where the tracker sees the ball. Alternating runs; longer gaps during stoppages."""
    n = len(labels)
    seen = np.zeros(n, dtype=bool)
    t, visible = 0, True
    while t < n:
        if visible:
            length = int(rng.geometric(1.0 / visible_mean))
        else:
            mean = dead_missing_mean if labels[t] == D else missing_mean
            length = int(rng.geometric(1.0 / mean))
        seen[t:t + length] = visible
        t += length
        visible = not visible
    return seen & (labels != OUT)


def assign_slots(present, team_of_player, n_slots, rng, piece_mean=900, gap_mean=30):
    """(n, n_players) slot index or -1. Tracks break into pieces and resume in a new slot.

    Slots are split between the teams in index order (first half home, second half away), and a
    slot is reused once it is free -- which is how 46 slots end up covering a whole match.
    """
    n, n_players = present.shape
    slot_of = np.full((n, n_players), -1, dtype=np.int16)
    home_slots = (n_slots + 1) // 2
    pools = {0: list(range(home_slots)), 1: list(range(home_slots, n_slots))}
    free_at = np.zeros(n_slots, dtype=np.int64)
    for player in range(n_players):
        pool = pools[int(team_of_player[player])]
        for start, stop, on in runs(present[:, player].astype(np.int8)):
            if not on:
                continue
            t = start
            while t < stop:
                length = int(rng.exponential(piece_mean)) + FPS
                end = min(t + length, stop)
                candidates = [s for s in pool if free_at[s] <= t]
                slot = int(rng.choice(candidates)) if candidates else int(rng.choice(pool))
                slot_of[t:end, player] = slot
                free_at[slot] = end + 1
                gap = int(rng.exponential(gap_mean)) if rng.random() < 0.4 else int(rng.integers(0, 4))
                t = end + gap
    return slot_of


def slot_arrays(positions, slot_of, n_slots):
    """Scatter per-player positions into per-slot arrays, NaN where a slot is empty."""
    n = len(positions)
    xy = np.full((n, n_slots, 2), np.nan)
    frames, players = np.nonzero(slot_of >= 0)
    xy[frames, slot_of[frames, players]] = positions[frames, players]
    return xy


def speed_mps(xy):
    """Speed from positions, in m/s. The real ATD writes m/s in channel 's' even though its metadata
    declares 'kph' -- reproduced here: the label says kph, the values are m/s."""
    metres = xy * np.array([LENGTH, WIDTH])
    step = np.full(xy.shape[:-1], np.nan)
    step[1:] = np.linalg.norm(np.diff(metres, axis=0), axis=-1)
    # Where the previous frame is missing, borrow the next step so a position never lacks a speed.
    first = np.isnan(step) & ~np.isnan(xy[..., 0])
    nxt = np.full_like(step, np.nan)
    nxt[:-1] = step[1:]
    step[first] = nxt[first]
    step[first & np.isnan(step)] = 0.0
    return step * FPS


# -- writers -----------------------------------------------------------------------------------

def write_atd_raw(path, xy, speed, ball_xy, ball_speed):
    """'<frame_id>:<x,y,s>;...;<x,y,s>:<x,y,s>' -- 5-decimal positions, full-precision speeds."""
    missing = "NaN,NaN,NaN"
    with open(path, "w", newline="\n") as handle:
        for i in range(len(xy)):
            groups = [missing if np.isnan(x) else "{:.5f},{:.5f},{!r}".format(x, y, float(s))
                      for (x, y), s in zip(xy[i], speed[i])]
            bx, by = ball_xy[i]
            ball = missing if np.isnan(bx) else "{:.5f},{:.5f},{!r}".format(
                bx, by, float(ball_speed[i]))
            handle.write("{}:{}:{}\n".format(i + 1, ";".join(groups), ball))


def sub(parent, tag, text=None, **attrib):
    element = ET.SubElement(parent, tag, attrib)
    if text is not None:
        element.text = str(text)
    return element


def params(parent, tag, pairs):
    group = sub(parent, tag)
    for name, value in pairs:
        param = sub(group, "ProviderParameter")
        sub(param, "Name", name)
        sub(param, "Value", value)


def write_atd_meta(path, n_frames, n_slots, halves):
    """FIFA/EPTS metadata with the same tag tree, key names and units as the real file."""
    root = ET.Element("main", {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                               "xsi:schemaLocation": "synthetic-schema-location"})
    meta = sub(root, "Metadata")
    config = sub(meta, "GlobalConfig")
    sub(config, "FileDate", "2000-01-01 00:00:00")
    sub(config, "FileName", "SYNTHETIC -- made-up values")
    sub(config, "TrackingType", "Optical")
    sub(config, "ProviderName", "Metrica Sports")
    sub(config, "FrameRate", FPS)
    bounds = {}
    for name, (first, last) in zip(("first_half", "second_half"), halves):
        bounds[name + "_start"], bounds[name + "_end"] = first + 1, last + 1  # frame_ids
    params(config, "ProviderGlobalParameters", [
        ("data_type", "synthetic"),
        ("first_half_start", bounds.get("first_half_start", "")),
        ("first_half_end", bounds.get("first_half_end", "")),
        ("second_half_start", bounds.get("second_half_start", "")),
        ("second_half_end", bounds.get("second_half_end", "")),
        ("first_extra_half_start", ""), ("first_extra_half_end", ""),
        ("second_extra_half_start", ""), ("second_extra_half_end", ""),
        ("tracking_source", "synthetic"), ("version", "0")])

    session = sub(sub(meta, "Sessions"), "Session", id="SYN_SESSION")
    sub(session, "SessionType", "Match")
    sub(session, "Start", "2000-01-01T00:00:00")
    score = sub(sub(session, "MatchParameters"), "Score",
                idLocalTeam=TEAM_IDS[0], idVisitingTeam=TEAM_IDS[1])
    sub(score, "LocalTeamScore", 0)
    sub(score, "VisitingTeamScore", 0)
    field = sub(session, "FieldSize")
    sub(field, "Width", 105)
    sub(field, "Height", 68)
    sub(session, "Competition", id="SYN")
    params(session, "ProviderSessionParameters", [("matchday", "0")])

    teams = sub(meta, "Teams")
    for team_id, name, direction in zip(TEAM_IDS, TEAM_NAMES, ("left_to_right", "right_to_left")):
        team = sub(teams, "Team", id=team_id)
        sub(team, "Name", name)
        params(team, "ProviderTeamsParameters", [("attack_direction_first_half", direction)])

    players = sub(meta, "Players")
    home_slots = (n_slots + 1) // 2
    for slot in range(1, n_slots + 1):
        player = sub(players, "Player", id=str(slot),
                     teamId=TEAM_IDS[0] if slot <= home_slots else TEAM_IDS[1])
        sub(player, "Name", "Track {}".format(slot))
        sub(player, "ShirtNumber", slot)
        params(player, "ProviderPlayerParameters", [
            ("position_index", -1), ("position_type", "Unknown"),
            ("position_x", 0), ("position_y", 0)])

    device = sub(sub(meta, "Devices"), "Device", id="device1")
    sub(device, "Name", "Device 1")
    sensors = sub(device, "Sensors")
    for sensor_id, sensor_name, channels in (
            ("position", "Position", (("x", "position_x", "normalized"),
                                      ("y", "position_y", "normalized"))),
            ("speed", "Speed", (("s", "speed", "kph"),))):
        sensor = sub(sensors, "Sensor", id=sensor_id)
        sub(sensor, "Name", sensor_name)
        channel_parent = sub(sensor, "Channels")
        for channel_id, channel_name, unit in channels:
            channel = sub(channel_parent, "Channel", id=channel_id)
            sub(channel, "Name", channel_name)
            sub(channel, "Unit", unit)

    player_channels = sub(meta, "PlayerChannels")
    for slot in range(1, n_slots + 1):
        for channel in "xys":
            sub(player_channels, "PlayerChannel", id="player{}_{}".format(slot, channel),
                playerId=str(slot), channelId=channel)

    spec = sub(sub(root, "DataFormatSpecifications"), "DataFormatSpecification",
               startFrame="1", endFrame=str(n_frames), separator=":")
    sub(spec, "StringRegister", name="frameCount")
    players_register = sub(spec, "SplitRegister", separator=";")
    for slot in range(1, n_slots + 1):
        register = sub(players_register, "SplitRegister", separator=",")
        for channel in "xys":
            sub(register, "PlayerChannelRef", playerChannelId="player{}_{}".format(slot, channel))
    ball_register = sub(spec, "SplitRegister", separator=",")
    for channel in "xys":
        sub(ball_register, "BallChannelRef", channelId=channel)

    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def homography_tactical(n, labels, rng):
    """(n, 3, 3) normalised pitch -> 1920x1080 pixels: a wide fixed camera that pans slightly."""
    import cv2
    base = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
        np.float32([[310, 187], [1612, 193], [1883, 958], [36, 962]])).astype(np.float64)
    pan = 40.0 * np.sin(2 * np.pi * np.arange(n) / (FPS * 90.0))
    shift = np.tile(np.eye(3), (n, 1, 1))
    shift[:, 0, 2] = pan
    # A fitted matrix is never exact: a little noise keeps every entry a non-round float.
    matrices = (shift @ base) * (1 + 1e-4 * rng.standard_normal((n, 3, 3)))
    matrices /= matrices[:, 2:3, 2:3]
    error = np.where(labels == OUT, rng.lognormal(np.log(1.5), 0.3, n),
                     rng.lognormal(np.log(0.24), 0.3, n))
    return matrices, error, np.ones(n, dtype=np.int64)


def write_homography(path, matrices, error, valid, h22_text):
    """frame,h00..h22,reprojection_error,valid -- %.7g, as in the real files.

    Tactical writes h22 as '1' (integer column); broadcast writes floats, and on invalid rows the
    nine matrix fields are the token 'nan' while reprojection_error stays a number.
    """
    header = "frame," + ",".join("h{}{}".format(r, c) for r in range(3) for c in range(3)) \
        + ",reprojection_error,valid\n"
    with open(path, "w", newline="\n") as handle:
        handle.write(header)
        for i in range(len(matrices)):
            if valid[i]:
                cells = ["{:.7g}".format(v) for v in matrices[i].ravel()[:8]] + [h22_text]
            else:
                cells = ["nan"] * 9
            handle.write("{},{},{:.7g},{}\n".format(i, ",".join(cells), error[i], int(valid[i])))


def broadcast_mapping(n_tactical, n_broadcast, start, rng, live_s=(15, 60), replay_p=0.3):
    """(tactical frame per broadcast frame, is_replay). Live stretches with cuts, plus replays."""
    frames, replay = [], []
    t = start
    while len(frames) < n_broadcast:
        live = int(rng.uniform(*live_s) * FPS)
        frames.extend(range(t, min(t + live, n_tactical)))
        replay.extend([False] * (min(t + live, n_tactical) - t))
        t = min(t + live + int(rng.uniform(0, 3) * FPS), n_tactical - 1)  # a cut skips a bit
        if rng.random() < replay_p:
            back = int(rng.uniform(5, 12) * FPS)
            frames.extend(range(max(0, t - back), t))
            replay.extend([True] * (t - max(0, t - back)))
    return np.array(frames[:n_broadcast]), np.array(replay[:n_broadcast])


def homography_broadcast(focus, replay, rng):
    """A zoomed camera following the play; invalid on replays and close-ups."""
    import cv2
    n = len(focus)
    matrices = np.full((n, 3, 3), np.nan)
    closeup = replay | (rng.random(n) < 0.0)  # replays only; the real file also has crowd shots
    for i in np.flatnonzero(~closeup):
        cx = focus[i]
        # Show ~40% of the pitch length around the focus, stretched to the frame width.
        x0, x1 = cx - 0.2, cx + 0.2
        src = np.float32([[x0, 0], [x1, 0], [x1, 1], [x0, 1]])
        dst = np.float32([[0, 120], [1920, 120], [2100, 1060], [-180, 1060]])
        matrices[i] = cv2.getPerspectiveTransform(src, dst)
        matrices[i] /= matrices[i, 2, 2]
    error = np.where(closeup, rng.uniform(5, 300, n), rng.lognormal(np.log(0.6), 0.5, n))
    return matrices, error, (~closeup).astype(np.int64)


def write_smart_tagging(path, labels, halves, kickoffs, rng):
    """SportsCode XML: <file><ALL_INSTANCES><instance>...</instance></ALL_INSTANCES><ROWS>."""
    instances = []

    def add(code, start_frame, end_frame, team, half, **extra):
        tags = [("Team", team), ("Half", half)] + list(extra.items())
        start_s = max(0.0, start_frame / FPS)
        instances.append((start_s, max(end_frame / FPS, start_s + 1.0), code, tags))

    for index, ((first, last), kickoff) in enumerate(zip(halves, kickoffs)):
        half = ("1st Half", "2nd Half")[index]
        add("SET PIECES", kickoff - 3.5 * FPS, kickoff + 1.5 * FPS, ST_TEAM[index % 2], half,
            Type="Kick Off (Start)")
        segments = [r for r in runs(labels[first:last + 1])]
        for position, (start, stop, label) in enumerate(segments):
            start, stop = start + first, stop + first
            if label in (H, A):
                if rng.random() < 0.75:
                    code = str(rng.choice(ATTACK_CODES))
                    extra = {}
                    if code == "PLAYERS IN THE BOX":
                        extra["Max Players in the box"] = str(rng.choice(["1", "2", "3", "4", "5",
                                                                          "6", "7+"]))
                    if code == "BALL IN THE BOX":
                        extra["Side"] = str(rng.choice(["Central", "Left", "Right"]))
                        extra["Direction of ball entry"] = str(rng.choice(
                            ["Horizontal", "Vertical", "Diagonal"]))
                    add(code, start + rng.uniform(0, 2) * FPS, stop + rng.uniform(-2, 3) * FPS,
                        ST_TEAM[label], half, **extra)
                if rng.random() < 0.4:
                    add(str(rng.choice(DEFEND_CODES)), start + rng.uniform(0, 3) * FPS,
                        stop + rng.uniform(-3, 2) * FPS, ST_TEAM[1 - label], half)
            elif label == D and position + 1 < len(segments) and start != first \
                    and rng.random() < 0.85:
                kind = str(rng.choice(SET_PIECE_TYPES))
                next_label = segments[position + 1][2]
                team = "N/A" if kind == "Throw-in" else ST_TEAM[next_label]
                add("SET PIECES", stop - 3.5 * FPS, stop + 1.5 * FPS, team, half, Type=kind)

    root = ET.Element("file")
    container = sub(root, "ALL_INSTANCES")
    for number, (start, end, code, tags) in enumerate(sorted(instances), start=1):
        instance = sub(container, "instance")
        sub(instance, "ID", number)
        sub(instance, "start", "{:.2f}".format(start))
        sub(instance, "end", "{:.2f}".format(end))
        sub(instance, "code", code)
        for group, text in tags:
            label = sub(instance, "label")
            sub(label, "group", group)
            sub(label, "text", text)
    rows = sub(root, "ROWS")
    for code in ALL_CODES:
        row = sub(rows, "row")
        sub(row, "code", code)
        for channel in "RGB":
            sub(row, channel, int(rng.integers(0, 65536)))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return len(instances)


def write_labels(path, labels):
    letters = np.array(["H", "A", "D"])
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        for frame, label in enumerate(labels):
            handle.write("{},{}\n".format(frame, "X" if label == OUT else letters[label]))


def write_video(path, matrices, positions, ball, until):
    """Tactical-looking mp4: pitch, all 22 players and the TRUE ball, frame number burned in.

    Only the first `until` frames are written: frame i of this file is video frame i, as in the
    real one, but drawing all 199,319 would take far longer than any test needs.
    """
    import cv2
    width, height = 1920, 1080
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    halfway = np.array([[0.5, 0], [0.5, 1]], float)
    colours = [(40, 40, 214)] * 11 + [(180, 119, 31)] * 11  # BGR: red home, blue away

    def project(matrix, points):
        homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
        return (homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.int32)

    for i in range(min(until, len(matrices))):
        image = np.full((height, width, 3), (40, 110, 40), np.uint8)
        cv2.fillConvexPoly(image, project(matrices[i], corners), (63, 143, 79))
        cv2.polylines(image, [project(matrices[i], corners)], True, (255, 255, 255), 2)
        cv2.polylines(image, [project(matrices[i], halfway)], False, (255, 255, 255), 2)
        if not np.isnan(positions[i, 0, 0]):
            for player, point in enumerate(project(matrices[i], positions[i])):
                cv2.circle(image, tuple(int(v) for v in point), 9, colours[player], -1)
        if not np.isnan(ball[i, 0]):
            point = project(matrices[i], ball[i:i + 1])[0]
            cv2.circle(image, tuple(int(v) for v in point), 6, (255, 255, 255), -1)
        cv2.putText(image, "SYNTHETIC  video frame {}".format(i), (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        writer.write(image)
    writer.release()


# -- main --------------------------------------------------------------------------------------

def build(out, full, video, video_until, seed):
    rng = np.random.default_rng(seed)
    n, halves, kickoffs = timeline(full)
    labels = simulate_labels(n, halves, kickoffs, rng)
    positions = simulate_positions(labels, halves, rng)
    in_match = labels != OUT
    positions[~in_match] = np.nan              # almost nobody tracked outside the halves
    ball = simulate_ball(labels, positions, rng)
    ball_speed = speed_mps(ball[:, None, :])[:, 0]
    team_of_player = np.repeat([0, 1], 11)

    # -- tactical: everyone in view, the ball seen ~57% of the time
    paths = asset_paths(data_dir=out, source="tactical")
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    slot_of = assign_slots(np.repeat(in_match[:, None], N_PLAYERS, axis=1), team_of_player,
                           N_SLOTS["tactical"], rng)
    xy = slot_arrays(positions, slot_of, N_SLOTS["tactical"])
    seen = visibility_mask(labels, rng)
    tracked_ball = np.where(seen[:, None], ball, np.nan)
    write_atd_meta(paths["atd_meta"], n, N_SLOTS["tactical"], halves)
    write_atd_raw(paths["atd_raw"], xy, speed_mps(xy), tracked_ball, ball_speed)
    matrices, error, valid = homography_tactical(n, labels, rng)
    write_homography(paths["homography"], matrices, error, valid, "1")
    n_tags = write_smart_tagging(paths["smart_tagging"], labels, halves, kickoffs, rng)
    print("tactical   {} frames, {} smart-tagging instances -> {}".format(
        n, n_tags, os.path.dirname(paths["atd_raw"])))
    if video:
        write_video(paths["video"], matrices, positions, ball, video_until)
        print("           video, first {} frames -> {}".format(video_until, paths["video"]))
    if full:
        write_labels(os.path.join(out, "BAR-ATH_synthetic_labels.csv"), labels)

    # -- broadcast: its own timeline (cuts, replays), only players near the play, 51 slots
    n_b = SOURCES["broadcast"]["n_video_frames"] if full else SMALL_BROADCAST
    if full:
        mapping, replay = broadcast_mapping(n, n_b, halves[0][0] - 500, rng)
    else:  # short enough that it needs a guaranteed replay to show the invalid rows
        mapping, replay = broadcast_mapping(n, n_b, halves[0][0], rng, (8, 12), 1.0)
    b_pos = positions[mapping].copy()
    known = ~np.isnan(b_pos[:, :, 0])
    focus = np.where(known.any(axis=1),
                     np.nansum(b_pos[:, :, 0], axis=1) / np.maximum(known.sum(axis=1), 1), 0.5)
    in_view = np.abs(b_pos[:, :, 0] - focus[:, None]) < 0.22
    in_view &= ~np.isnan(b_pos[:, :, 0]) & ~replay[:, None]
    b_paths = asset_paths(data_dir=out, source="broadcast")
    os.makedirs(os.path.dirname(b_paths["atd_raw"]), exist_ok=True)
    b_slot = assign_slots(in_view, team_of_player, N_SLOTS["broadcast"], rng, piece_mean=300)
    b_xy = slot_arrays(b_pos, b_slot, N_SLOTS["broadcast"])
    b_seen = visibility_mask(labels[mapping], rng, visible_mean=50, missing_mean=45) & ~replay
    b_ball = np.where(b_seen[:, None], ball[mapping], np.nan)
    write_atd_meta(b_paths["atd_meta"], n_b, N_SLOTS["broadcast"], halves)
    write_atd_raw(b_paths["atd_raw"], b_xy, speed_mps(b_xy), b_ball,
                  speed_mps(ball[mapping][:, None, :])[:, 0])
    b_matrices, b_error, b_valid = homography_broadcast(focus, replay, rng)
    write_homography(b_paths["homography"], b_matrices, b_error, b_valid, "1.0")
    print("broadcast  {} frames, {:.1f}% valid homography -> {}".format(
        n_b, 100 * b_valid.mean(), os.path.dirname(b_paths["atd_raw"])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=HERE)
    parser.add_argument("--full", action="store_true", help="full-length bundle (~400 MB)")
    parser.add_argument("--video", action="store_true", help="also write a synthetic mp4")
    parser.add_argument("--video-until", type=int, default=19000,
                        help="number of video frames to write (default covers the public "
                             "5-minute window)")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    build(os.path.abspath(args.out), args.full, args.video, args.video_until, args.seed)
