"""Download the Sportec Open DFL dataset and convert it to the shared per-match format.

Seven professional matches (2 Bundesliga, 5 2. Bundesliga, 2022/23), optical tracking at 25 Hz
with a per-frame ball-possession and ball-status flag -- the same kind of label our ground truth is
derived from. Public, CC-BY 4.0:

    Bassek, Rein, Weber & Memmert (2025). An integrated dataset of spatiotemporal and event data in
    elite soccer. Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y
    Files: https://doi.org/10.6084/m9.figshare.28196177.v1

    python external/fetch_dfl.py                       # everything: ~2.6 GB raw, ~0.3 GB processed
    python external/fetch_dfl.py --matches J03WOH      # one match, for a quick test
    python external/fetch_dfl.py --no-convert          # download only
    python external/fetch_dfl.py --delete-raw          # keep only the processed files

Re-running is cheap: verified downloads and up-to-date conversions are skipped, and an interrupted
download resumes where it stopped. Output goes to external/dfl/ (or $DFL_DATA_DIR):

    raw/        the original XMLs, checked against figshare's MD5
    processed/  one parquet + JSON sidecar per match, in the format defined in schema.py

Labels: H / A from BallPossession (1 home, 2 guest) where BallStatus is 1, D where it is 0. These
are DFL's own flags, as delivered -- note that DFL switches possession on far shorter touches than
our ground truth does (see external/README.md), so any "controlled possession" filtering belongs in
the training code, not here.
"""

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from external import schema  # noqa: E402

# Params
FIGSHARE = "https://ndownloader.figshare.com/files/{}"
# The seven matches: {match id: {kind: (figshare file id, md5, file name)}}. Pinned rather than
# listed live, so every teammate gets byte-identical inputs; the MD5s are figshare's own.
_COM1, _COM2 = "DFL-COM-000001", "DFL-COM-000002"
FILES = {
    "J03WMX": {"info": (51643475, "bc5d5b21dbcf36aa30f4cd0428309f6b", _COM1),
               "events": (51643493, "2d88e180a1f63c7e4157feb980fc7255", _COM1),
               "positions": (51643514, "e4273a2adf5dbd9d4ec42def2f01905c", _COM1)},
    "J03WN1": {"info": (51643472, "37b0f85588c0acfb995fe633c7ac2e0d", _COM1),
               "events": (51643496, "83e8ed7ea8e8ebbbdc61f61ac3534008", _COM1),
               "positions": (51643517, "bacb2f609c1d1b0bfc906f6b9d4109f2", _COM1)},
    "J03WOH": {"info": (51643478, "b9710781d400dd812f62b0a6e3d7bce8", _COM2),
               "events": (51643499, "ab66e51512f39d06f49f27dd3b4c2f40", _COM2),
               "positions": (51643520, "3efff25a86a61768e7d253b4bba137a3", _COM2)},
    "J03WOY": {"info": (51643481, "7b021e577fca917a63111d55cc29ae5b", _COM2),
               "events": (51643502, "64b1be01d6ff6848168efebd56f344b9", _COM2),
               "positions": (51643523, "4e1063e0a0ae499920cf68ac1aa3ab4e", _COM2)},
    "J03WPY": {"info": (51643487, "65a4568e4f374af94d3c1015cbc7c4f5", _COM2),
               "events": (51643505, "f68482441d38723d69f244f896482803", _COM2),
               "positions": (51643526, "e56ff1336a334f8da51110fbe7cab764", _COM2)},
    "J03WQQ": {"info": (51643484, "5979209af316a914ce57347d0633df6c", _COM2),
               "events": (51643508, "b7e9d82af4034539d802319be6c57776", _COM2),
               "positions": (51643529, "9105bad987191d3c3c362c1fce9a4fcd", _COM2)},
    "J03WR9": {"info": (51643490, "5e2d580d3d8ed97e32959de90d04992b", _COM2),
               "events": (51643511, "c13bc5efa5eca4a022cc678210b87d5f", _COM2),
               "positions": (51643532, "e060bb7d7a5a3bfbd15bbf9bc631024d", _COM2)},
}
FILE_STEMS = {"info": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
              "positions": "DFL_04_03_positions_raw_observed"}
PERIODS = {"firstHalf": 1, "secondHalf": 2}
CHUNK = 1 << 20
# Bump when the conversion changes, so existing processed files are rebuilt on the next run.
CONVERTER_VERSION = 1

_ATTR = re.compile(r'(\w+)="([^"]*)"')
_PLAYER_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')


def file_name(match, kind):
    return "{}_{}_DFL-MAT-{}.xml".format(FILE_STEMS[kind], FILES[match][kind][2], match)


# -- download --------------------------------------------------------------------------------------

def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(file_id, md5, path):
    """Fetch one file into path, resuming a .part file if present. Verifies the MD5."""
    if os.path.isfile(path) and md5_of(path) == md5:
        return "ok"
    part = path + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    request = urllib.request.Request(FIGSHARE.format(file_id),
                                     headers={"Range": "bytes={}-".format(have)} if have else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:      # server ignored the range: start over
            have = 0
        total = have + int(response.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as handle:
            done, last = have, 0.0
            for block in iter(lambda: response.read(CHUNK), b""):
                handle.write(block)
                done += len(block)
                if total > 50 * CHUNK and time.time() - last > 2:
                    print("\r    {:5.1f}% of {:.0f} MB".format(100 * done / total, total / 2 ** 20),
                          end="", flush=True)
                    last = time.time()
        if total > 50 * CHUNK:
            print()
    got = md5_of(part)
    if got != md5:
        os.remove(part)
        raise IOError("{}: MD5 {} != {} -- deleted, run again".format(path, got, md5))
    os.replace(part, path)
    return "downloaded"


# -- parse -----------------------------------------------------------------------------------------

def read_match_info(path):
    """Teams (home first), slot candidates and pitch size from the match-information XML."""
    root = ET.parse(path).getroot()
    info = root.find("MatchInformation")
    general, env = info.find("General").attrib, info.find("Environment").attrib
    teams = {}
    for team in info.iter("Team"):
        players = {p.get("PersonId"): {"person_id": p.get("PersonId"),
                                       "shirt": p.get("ShirtNumber"),
                                       "position": p.get("PlayingPosition"),
                                       "starting": p.get("Starting") == "true"}
                   for p in team.iter("Player")}
        teams[team.get("Role")] = {"id": team.get("TeamId"), "name": team.get("TeamName"),
                                   "players": players}
    return {"match_id": general["MatchId"], "competition": general["CompetitionName"],
            "season": general["Season"], "matchday": general.get("MatchDay"),
            "title": general.get("MatchTitle"), "result": general.get("Result"),
            "home": teams["home"], "away": teams["guest"],
            "pitch": (float(env["PitchX"]), float(env["PitchY"]))}


def read_positions(path):
    """Every frameset of the positions XML, streamed: {(period, team, person): arrays}.

    Players carry N, X, Y; the ball also Z, BallPossession and BallStatus. ~1 minute per match.
    """
    sets, current, rows = {}, None, None

    def close():
        if current is not None and rows:
            sets[current] = {k: np.array(v) for k, v in rows.items()}

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if current[1] == "BALL":
                    attrs = dict(_ATTR.findall(line))
                    for key in ("N", "X", "Y", "Z", "BallPossession", "BallStatus"):
                        rows[key].append(attrs.get(key, "nan"))
                else:
                    found = _PLAYER_FRAME.match(line)
                    rows["N"].append(found.group(1))
                    rows["X"].append(found.group(2))
                    rows["Y"].append(found.group(3))
            elif line.startswith("<FrameSet "):
                close()
                attrs = dict(_ATTR.findall(line))
                current = (PERIODS[attrs["GameSection"]], attrs["TeamId"], attrs["PersonId"])
                keys = ("N", "X", "Y", "Z", "BallPossession", "BallStatus") \
                    if attrs["TeamId"] == "BALL" else ("N", "X", "Y")
                rows = {k: [] for k in keys}
    close()
    for arrays in sets.values():
        arrays["N"] = arrays["N"].astype(np.int64)
        for key in ("X", "Y", "Z"):
            if key in arrays:
                arrays[key] = arrays[key].astype(np.float32)
        for key in ("BallPossession", "BallStatus"):
            if key in arrays:
                arrays[key] = np.where(arrays[key] == "nan", "0", arrays[key]).astype(np.int8)
    return sets


# -- convert ---------------------------------------------------------------------------------------

def to_canonical(info, sets):
    """(table, meta) in the schema.py format from the parsed match info and framesets."""
    balls = {period: arrays for (period, team, _), arrays in sets.items() if team == "BALL"}
    if sorted(balls) != [1, 2]:
        raise ValueError("expected a ball frameset for both halves, got {}".format(sorted(balls)))

    # The ball is tracked on every frame of a half, so its frames are the timeline.
    timeline = []
    for period in (1, 2):
        frames = balls[period]["N"]
        if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
            raise ValueError("half {}: ball frames are not contiguous".format(period))
        timeline.append((period, int(frames[0]), len(frames)))
    n = sum(length for _, _, length in timeline)
    offset = {period: start for period, start, _ in timeline}
    base = {1: 0, 2: timeline[0][2]}

    # Slots: every player who appears, in order of first appearance (starters first, naturally).
    first_seen = {}
    for (period, team, person), arrays in sets.items():
        if team != "BALL":
            row = base[period] + int(arrays["N"][0]) - offset[period]
            first_seen[person] = min(first_seen.get(person, n), row)
    slots = {}
    for side in ("home", "away"):
        team_id = info[side]["id"]
        people = {person for (_, team, person) in sets if team == team_id}
        slots[side] = sorted(people, key=lambda person: (first_seen[person], person))

    frame = np.concatenate([balls[p]["N"] for p in (1, 2)])
    period = np.concatenate([np.full(len(balls[p]["N"]), p, dtype=np.int8) for p in (1, 2)])
    columns = {"frame": frame, "period": period}
    for axis in ("X", "Y", "Z"):
        columns["ball_" + axis.lower()] = np.concatenate([balls[p][axis] for p in (1, 2)])
    for side, prefix in (("home", "h"), ("away", "a")):
        team_id = info[side]["id"]
        for k, person in enumerate(slots[side], 1):
            xy = np.full((n, 2), np.nan, dtype=np.float32)
            for p in (1, 2):
                arrays = sets.get((p, team_id, person))
                if arrays is None:
                    continue
                rows = base[p] + arrays["N"] - offset[p]
                keep = (rows >= base[p]) & (rows < base[p] + len(balls[p]["N"]))
                xy[rows[keep], 0] = arrays["X"][keep]
                xy[rows[keep], 1] = arrays["Y"][keep]
            columns["{}{}_x".format(prefix, k)] = xy[:, 0]
            columns["{}{}_y".format(prefix, k)] = xy[:, 1]

    status = np.concatenate([balls[p]["BallStatus"] for p in (1, 2)])
    owner = np.concatenate([balls[p]["BallPossession"] for p in (1, 2)])
    label = np.where(status == 0, "D", np.where(owner == 1, "H", np.where(owner == 2, "A", "X")))
    columns["label"] = label
    table = pd.DataFrame(columns)

    # Rotate each half so that home attacks +x, judged from the kickoff positions.
    n_home, n_away = len(slots["home"]), len(slots["away"])
    rotated = []
    window = schema.ORIENTATION_WINDOW_S * schema.FPS
    for p in (1, 2):
        start = base[p]
        home = table[schema.player_columns("h", n_home)].to_numpy()[start:start + window]
        away = table[schema.player_columns("a", n_away)].to_numpy()[start:start + window]
        attacks_positive = schema.home_attacks_positive(home.reshape(len(home), n_home, 2),
                                                        away.reshape(len(away), n_away, 2))
        if attacks_positive is None:
            raise ValueError("half {}: cannot tell the direction of play at kickoff".format(p))
        if not attacks_positive:
            schema.rotate_180(table, n_home, n_away, table.index[table["period"] == p])
            rotated.append(p)

    meta = {
        "source": "dfl", "converter_version": CONVERTER_VERSION,
        "match_id": info["match_id"], "competition": info["competition"],
        "season": info["season"], "matchday": info["matchday"], "title": info["title"],
        "result": info["result"], "pitch": list(info["pitch"]), "fps": schema.FPS,
        "home_team": info["home"]["name"], "away_team": info["away"]["name"],
        "n_home": n_home, "n_away": n_away,
        "home_slots": [info["home"]["players"].get(p, {"person_id": p}) for p in slots["home"]],
        "away_slots": [info["away"]["players"].get(p, {"person_id": p}) for p in slots["away"]],
        "rotated_periods": rotated,
        "labels": "DFL BallStatus (0 -> D) and BallPossession (1 -> H, 2 -> A)",
        "license": "CC-BY 4.0, Bassek et al. 2025, doi:10.1038/s41597-025-04505-y",
    }
    return table, meta


def convert(match, raw_dir, out_dir, force=False):
    """Raw XMLs of one match -> processed/<DFL-MAT-id>.parquet. Skipped if already current."""
    name = "DFL-MAT-" + match
    sidecar = os.path.join(out_dir, name + ".json")
    if not force and os.path.isfile(sidecar):
        import json
        with open(sidecar) as handle:
            meta = json.load(handle)
        if (meta.get("converter_version") == CONVERTER_VERSION
                and meta.get("schema_version") == schema.SCHEMA_VERSION
                and os.path.isfile(os.path.join(out_dir, name + ".parquet"))):
            return "ok"
    info = read_match_info(os.path.join(raw_dir, file_name(match, "info")))
    sets = read_positions(os.path.join(raw_dir, file_name(match, "positions")))
    table, meta = to_canonical(info, sets)
    schema.write_match(table, meta, out_dir, name)
    return "converted"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matches", nargs="+", choices=sorted(FILES), default=sorted(FILES),
                        metavar="ID", help="match ids, e.g. J03WOH (default: all seven)")
    parser.add_argument("--out", default=schema.DFL_DIR,
                        help="root folder (default: external/dfl or $DFL_DATA_DIR)")
    parser.add_argument("--no-convert", action="store_true", help="download only")
    parser.add_argument("--force", action="store_true", help="re-convert even if up to date")
    parser.add_argument("--delete-raw", action="store_true",
                        help="delete each positions XML once its match is converted")
    args = parser.parse_args(argv)

    raw_dir = os.path.join(args.out, "raw")
    out_dir = os.path.join(args.out, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    for match in args.matches:
        print("DFL-MAT-{}".format(match))
        processed = os.path.join(out_dir, "DFL-MAT-{}.parquet".format(match))
        for kind in ("info", "events", "positions"):
            if kind == "positions" and args.delete_raw and os.path.isfile(processed) \
                    and not args.force:
                continue                  # converted earlier and its raw file deliberately gone
            file_id, md5, _ = FILES[match][kind]
            status = download(file_id, md5, os.path.join(raw_dir, file_name(match, kind)))
            print("  {:9} {}".format(kind, status))
        if args.no_convert:
            continue
        started = time.time()
        status = convert(match, raw_dir, out_dir, force=args.force)
        print("  {:9} {} ({:.0f} s)".format("convert", status, time.time() - started))
        if args.delete_raw:
            positions = os.path.join(raw_dir, file_name(match, "positions"))
            if os.path.isfile(positions):
                os.remove(positions)
    print("\nprocessed files in {}".format(out_dir))


if __name__ == "__main__":
    main()
