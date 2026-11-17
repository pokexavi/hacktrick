"""Convert our BAR-ATH tactical tracking (ATD) into the shared per-match format of schema.py.

    python external/convert_atd.py                          # data/ -> data/processed/BAR-ATH.*
    python external/convert_atd.py --data-dir cache/synthetic_full --labels <frame,label csv>

The output stays inside the gated data directory ($MTAH_DATA_DIR/processed), because it is
derived match data and the competition terms cover it like the original. It is gitignored there.

What changes on the way:
  - normalised 0-1 coordinates (origin top-left, y down)  ->  metres, centre origin, y up
  - each half rotated so that home attacks +x, from the ATD metadata, cross-checked against the
    kickoff positions
  - track slots split into home (h1..) and away (a1..) slots by the metadata's team assignment
  - positions further than schema's margin off the pitch -- tracker detections in the stands --
    set to NaN, and counted
  - labels: the public 5-minute ground truth by default, X everywhere else

Every row of the output is one video frame, so row i is still submission row i.

Prints counts and booleans only, never values: the match data must not reach a hosted LLM.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exploration"))
from external import schema  # noqa: E402

# Params
NAME = "BAR-ATH"
SOURCE = "tactical"
SECOND_HALF_KICKOFF = 113742          # video frame; FAQ.md. The 2nd half is graded from 110808
OFF_PITCH_MARGIN_M = 12.0             # beyond this outside the lines a "player" is not on the pitch
CONVERTER_VERSION = 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="bundle to read (default: $MTAH_DATA_DIR or ./data)")
    parser.add_argument("--labels", help="frame,label CSV for the label column "
                                         "(default: the public ground truth)")
    parser.add_argument("--refresh", action="store_true", help="re-parse the ATD, ignoring cache/")
    args = parser.parse_args(argv)
    if args.data_dir:
        os.environ["MTAH_DATA_DIR"] = os.path.abspath(args.data_dir)

    import common           # after MTAH_DATA_DIR is final: it decides the cache folder too
    import games

    trk = common.load_tracking(SOURCE, refresh=args.refresh)
    meta = trk.meta
    length, width = games.GAMES[games.GAME]["pitch"]
    n = len(trk.frame)

    # Metres, centre origin, y up.
    scale = np.array([length, -width], dtype=np.float32)
    xy = (trk.xy - 0.5) * scale
    ball = (trk.ball_xy - 0.5) * scale

    # Tracker detections far off the pitch are not players on it.
    limit = np.array([length / 2 + OFF_PITCH_MARGIN_M, width / 2 + OFF_PITCH_MARGIN_M])
    off = np.isfinite(xy[..., 0]) & (np.abs(xy) > limit).any(-1)
    xy[off] = np.nan
    ball_off = np.isfinite(ball[:, 0]) & (np.abs(ball) > limit).any(-1)
    ball[ball_off] = np.nan

    period = np.zeros(n, dtype=np.int8)
    for number, first, last in games.half_video_blocks(games.GAME):
        period[(trk.frame >= first) & (trk.frame <= last)] = number

    # Orientation per half. The metadata names the home team's first-half direction; the second
    # half is the opposite. Frames outside the halves follow the half they sit next to.
    home_dir = meta["attack_direction_first_half"][0]
    if home_dir not in ("left_to_right", "right_to_left"):
        raise ValueError("unexpected attack_direction_first_half: {!r}".format(home_dir))
    first_half_positive = home_dir == "left_to_right"
    second_half_start = games.half_video_blocks(games.GAME)[1][1]
    in_second = trk.frame >= second_half_start
    rotate = np.where(in_second, first_half_positive, not first_half_positive)
    xy[rotate] *= -1
    ball[rotate] *= -1

    home_slots = np.flatnonzero(trk.slot_team == common.HOME)
    away_slots = np.flatnonzero(trk.slot_team == common.AWAY)
    window = schema.ORIENTATION_WINDOW_S * schema.FPS
    kickoffs = {1: games.half_video_blocks(games.GAME)[0][1], 2: SECOND_HALF_KICKOFF}
    agrees = {}
    for half, kickoff in kickoffs.items():
        rows = np.flatnonzero((trk.frame >= kickoff) & (trk.frame < kickoff + window))
        agrees[half] = schema.home_attacks_positive(xy[rows][:, home_slots], xy[rows][:, away_slots])

    if args.labels:
        from possession_metrics import read_label_csv
        labels = np.array([str(v).strip().upper() for v in read_label_csv(args.labels)["label"]])
    else:
        labels = common.load_gt_public()
    label = np.where(trk.frame < len(labels), labels[np.minimum(trk.frame, len(labels) - 1)], "X")
    label = np.where(period == 0, "X", label)

    columns = {"frame": trk.frame.astype(np.int64), "period": period,
               "ball_x": ball[:, 0], "ball_y": ball[:, 1],
               "ball_z": np.full(n, np.nan, dtype=np.float32)}
    for prefix, slots in (("h", home_slots), ("a", away_slots)):
        for k, slot in enumerate(slots, 1):
            columns["{}{}_x".format(prefix, k)] = xy[:, slot, 0]
            columns["{}{}_y".format(prefix, k)] = xy[:, slot, 1]
    columns["label"] = label
    table = pd.DataFrame(columns)

    out_meta = {
        "source": "bar", "converter_version": CONVERTER_VERSION, "match_id": NAME,
        "competition": "LaLiga EA Sports", "season": "2024/2025", "matchday": "2",
        "title": "{}:{}".format(*meta["team_names"]), "pitch": [length, width],
        "fps": schema.FPS, "home_team": meta["team_names"][0], "away_team": meta["team_names"][1],
        "n_home": int(len(home_slots)), "n_away": int(len(away_slots)),
        "home_slots": [{"slot_id": meta["slot_ids"][s]} for s in home_slots],
        "away_slots": [{"slot_id": meta["slot_ids"][s]} for s in away_slots],
        "rotated_periods": [h for h in (1, 2) if (rotate & (period == h)).any()],
        "labels": os.path.basename(args.labels) if args.labels else "public ground truth (5 min)",
        "tracking_source": SOURCE, "data_dir": common.data_dir(),
    }
    path = schema.write_match(table, out_meta, schema.bar_processed_dir(), NAME)

    graded = period > 0
    print("wrote {}".format(path))
    print("  rows {}  (= video frames: {})".format(n, n == games.source_frames(SOURCE)))
    print("  slots: home {}, away {}".format(len(home_slots), len(away_slots)))
    print("  rotated halves: {}".format(out_meta["rotated_periods"]))
    print("  kickoff positions agree with metadata direction: half 1 {}, half 2 {}".format(
        agrees[1], agrees[2]))
    print("  off-pitch detections set to NaN: players {}, ball {}".format(int(off.sum()),
                                                                         int(ball_off.sum())))
    print("  ball tracked on {:.1%} of graded frames".format(np.isfinite(ball[graded, 0]).mean()))
    print("  labelled frames: {}".format(int((label != "X").sum())))
    if not all(agrees.values()):
        print("\n  WARNING: the kickoff positions disagree with the metadata direction. Check the "
              "orientation before training on this file.")


if __name__ == "__main__":
    main()
