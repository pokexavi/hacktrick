"""
How BAR-ATH's Smart Tagging set pieces line up with a model's stoppages, without labels.

    python -m dead_alive_detection.analysis.tag_alignment

Saves per-class aggregates to results/<out>/tag_alignment.json, which decoding/tags.py reads to
simulate the tags on DFL.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from starter.load_smart_tagging import load  # noqa: E402

KINDS = {"Throw-in": "throw_in", "Corner Kick": "corner", "Goal Kick": "goal_kick",
         "Free Kick": "free_kick", "Kick Off (After Goal)": "kickoff"}
FPS = 25


def set_pieces():
    """
    BAR-ATH set-piece tags of the kinds we can simulate, as {kind, start, stop} in frames.
    """
    return [{"kind": KINDS[e["tags"].get("Type")], "start": e["start_frame"], "stop": e["end_frame"]}
            for e in load() if (e["code"] or "").strip().lower() == "set pieces"
            and e["tags"].get("Type") in KINDS]


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ens_final3_all7", help="folder of cache/dead_alive")
    parser.add_argument("--out", default="ens_final3", help="results/<out>/tag_alignment.json")
    args = parser.parse_args(argv)

    stops = pd.read_csv(os.path.join(ROOT, "cache", "dead_alive", args.model, "intervals.csv"))
    tags = set_pieces()
    rows = []
    for tag in tags:
        inside = stops[(stops["start"] < tag["stop"]) & (stops["stop"] > tag["start"])]
        row = {"kind": tag["kind"], "window_s": (tag["stop"] - tag["start"]) / FPS, "n_inside": len(inside)}
        if len(inside):
            best = inside.iloc[np.argmax([overlap(tag["start"], tag["stop"], a, b)
                                          for a, b in zip(inside["start"], inside["stop"])])]
            row.update({"end_in_s": (best["stop"] - tag["start"]) / FPS,
                        "start_in_s": (best["start"] - tag["start"]) / FPS,
                        "stop_len_s": (best["stop"] - best["start"]) / FPS})
        rows.append(row)
    table = pd.DataFrame(rows)

    def counts(frame):
        return {"n": len(frame), "no_stoppage": int((frame.n_inside == 0).sum()),
                "window_s": [float(x) for x in frame.window_s.round(1)],
                "end_in_s": [float(x) for x in frame.end_in_s.dropna().round(1)],
                "start_in_s": [float(x) for x in frame.start_in_s.dropna().round(1)],
                "n_inside_mean": float(frame.n_inside.mean())}

    summary = {kind: counts(part) for kind, part in table.groupby("kind")}
    summary["all"] = counts(table)
    n_stops = len(stops)
    tagged = sum(any(overlap(a, b, t["start"], t["stop"]) > 0 for t in tags)
                 for a, b in zip(stops["start"], stops["stop"]))
    summary["stoppages"] = {"n": int(n_stops), "with_a_tag": int(tagged)}
    if "type" in stops:
        summary["model_class_share_with_tag"] = {
            kind: {"n": int(len(p)), "with_tag": int(sum(any(overlap(a, b, t["start"], t["stop"]) > 0 for t in tags)
                                                         for a, b in zip(p["start"], p["stop"])))}
            for kind, p in stops.groupby("type")}

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", args.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tag_alignment.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    desc = table.groupby("kind").agg(n=("window_s", "size"), window_s=("window_s", "median"),
                                     no_stoppage=("n_inside", lambda s: int((s == 0).sum())),
                                     n_inside=("n_inside", "mean"), end_in_s_med=("end_in_s", "median"),
                                     end_in_s_q10=("end_in_s", lambda s: s.quantile(.1)),
                                     end_in_s_q90=("end_in_s", lambda s: s.quantile(.9)),
                                     start_in_s_med=("start_in_s", "median"))
    print(desc.to_string(float_format="{:.1f}".format))
    print("all: {} tags, {} with no stoppage; stoppages of the model with a tag: {}/{}".format(
        len(table), int((table.n_inside == 0).sum()), tagged, n_stops))
    if "model_class_share_with_tag" in summary:
        print(json.dumps(summary["model_class_share_with_tag"]))


if __name__ == "__main__":
    main()
