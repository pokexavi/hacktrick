"""
Build a BAR-ATH model folder for the Dead-ball review page from a run's saved heads.

    python -m dead_alive_detection.predict --run results/tcn_bar_1

No training: it decodes the heads a run already saved, so other decoders can be tried on BAR. Writes
frames, intervals, decisions and a submission to cache/dead_alive/<name>/.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from dead_alive_detection import config as config_lib
from dead_alive_detection.loading.clean import played
from dead_alive_detection.loading.tensors import to_frames
from external import schema

DEAD_ALIVE = os.path.join(schema.ROOT, "cache", "dead_alive")


def frame_table(match, rows, dead, p):
    """
    frames.csv table: dead per frame, with the per-step p spread back to every frame.
    """
    n = len(match.frame)
    in_half = match.period > 0
    p_frames = np.where(in_half, to_frames(p, rows, n), np.nan)
    state = np.where(in_half, np.where(dead, "D", "alive"), "X")
    return pd.DataFrame({"frame": match.frame, "p_dead_frame": p_frames, "p_dead": p_frames,
                         "state": state, "ball_observed": np.isfinite(match.ball[:, 0]).astype(int)})


def intervals(frames, period):
    """
    intervals.csv table: one row per predicted stoppage, never crossing halftime.
    """
    dead = frames["state"].to_numpy() == "D"
    p = frames["p_dead"].to_numpy()
    change = np.flatnonzero((dead[1:] != dead[:-1]) | (period[1:] != period[:-1])) + 1
    rows = [{"start": int(frames["frame"].iloc[a]), "stop": int(frames["frame"].iloc[b - 1]) + 1,
             "seconds": (b - a) / schema.FPS, "half": int(period[a]),
             "mean_p_dead": round(float(p[a:b].mean()), 3),
             "peak_p_dead": round(float(p[a:b].max()), 3),
             "mean_p_dead_frame": round(float(p[a:b].mean()), 3)}
            for a, b in zip(np.r_[0, change], np.r_[change, len(dead)]) if dead[a]]
    return pd.DataFrame(rows, columns=["start", "stop", "seconds", "half", "mean_p_dead",
                                       "peak_p_dead", "mean_p_dead_frame"])


def write_folder(out, match, bar, decoder, cfg, info):
    """
    Decode the saved BAR heads and write the model folder.
    """
    from dead_alive_detection.decoding.decoder import type_probabilities
    from dead_alive_detection.submit import build, read_labels
    # Decode only the steps inside the match's halves: heads saved before played() also cover
    # BAR's second half before its kickoff, which is not labelled (submit.build fills it with D).
    bar = {**bar, "period": np.where(match.period[bar["row"]] > 0, bar["period"], 0)}
    dead_steps, kind = decoder.decode_types(bar["logits"], bar["prior"], bar["period"])
    n = len(match.frame)
    p = 1 / (1 + np.exp(-bar["logits"][:, 0]))
    frames = frame_table(match, bar["row"], to_frames(dead_steps, bar["row"], n), p)
    # The start and stop heads (peaks where a stoppage begins / play restarts), for the app.
    for k, column in ((1, "p_start"), (2, "p_stop")):
        q = 1 / (1 + np.exp(-bar["logits"][:, k]))
        frames[column] = np.where(match.period > 0, to_frames(q, bar["row"], n), np.nan)
    classes = cfg["types"]["classes"]
    if classes and bar["logits"].shape[1] > 3:
        # The stoppage class: dead_type per frame, its probabilities, a type per stoppage.
        kind = to_frames(kind, bar["row"], n)
        frames["dead_type"] = np.where((frames["state"] == "D") & (kind >= 0),
                                       np.array(classes, dtype=object)[np.maximum(kind, 0)], "")
        probs = type_probabilities(bar["logits"])
        for k, c in enumerate(classes):
            frames["p_" + c] = np.where(match.period > 0, to_frames(probs[:, k], bar["row"], n), np.nan)
    os.makedirs(out, exist_ok=True)
    frames.to_csv(os.path.join(out, "frames.csv"), index=False, float_format="%.4g")
    stops = intervals(frames, match.period)
    if "dead_type" in frames and len(stops):
        stops["type"] = frames["dead_type"].to_numpy()[stops["start"].to_numpy()]
        stops["type_p"] = [round(float(frames["p_" + k].to_numpy()[a:b].mean()), 3)
                           for k, a, b in zip(stops["type"], stops["start"], stops["stop"])]
    stops.to_csv(os.path.join(out, "intervals.csv"), index=False)
    from dead_alive_detection.decoding import explain
    explain.write(out, decoder, bar)
    teams = read_labels(config_lib.resolve_path(cfg["bar"]["teams"]))
    if len(teams) == n:
        labels, _ = build(frames, teams)
        pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(
            os.path.join(out, "submission.csv"), index=False)
    else:
        print("no submission.csv: [bar] teams does not have one row per frame of this match")
    with open(os.path.join(out, "model.json"), "w") as handle:
        json.dump({"model": "TCN + HSMM (dead_alive_detection)", **info,
                   **dict(zip(("temperature", "w_start", "w_stop", "type_weight", "type_evidence"),
                              decoder.setting))}, handle)
    config_lib.dump(cfg, os.path.join(out, "config.toml"))
    graded = frames["state"] != "X"
    print("wrote {}: {} predicted stoppages, {:.1%} of the halves dead".format(
        out, len(stops), (frames.loc[graded, "state"] == "D").mean()))
    return out


def main(argv=None):
    from dead_alive_detection.decoding import grid
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="results/<run> folder")
    parser.add_argument("--oof", help="folder with bar.npz (default: the run's)")
    parser.add_argument("--config", help="config (default: the run's)")
    parser.add_argument("--name", help="cache/dead_alive/<name> (default: [bar] name, else the run)")
    parser.add_argument("--decoder", choices=("config", "grid"),
                        help="override [bar] decoder; grid needs the run's grid.csv")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if not args.run and not (args.oof and args.config):
        parser.error("give --run, or --oof and --config")
    cfg = config_lib.load(args.config or os.path.join(args.run, "config.toml"), args.set)
    run_name = os.path.basename(os.path.normpath(args.run or args.oof))
    folder = args.oof or os.path.join(config_lib.resolve_path(cfg["runtime"]["cache"]), run_name)
    bar = dict(np.load(os.path.join(folder, "bar.npz")))
    setting = None
    if (args.decoder or cfg["bar"]["decoder"]) == "grid":
        setting = grid.choice(grid.read_grid(os.path.join(args.run, "grid.csv")), cfg)
    decoder = grid.decoder_for(list(bar["train"]), cfg, setting)
    match = played(schema.load_match(config_lib.resolve_path(cfg["bar"]["match"])))
    name = args.name or cfg["bar"]["name"] or run_name
    write_folder(os.path.join(DEAD_ALIVE, name), match, bar, decoder, cfg,
                 {"heads": os.path.relpath(folder, schema.ROOT).replace("\\", "/")})


if __name__ == "__main__":
    main()
