"""
Compares BAR-ATH model folders (cache/dead_alive/<model>/) side by side.

Each folder holds the frames.csv, intervals.csv and submission.csv written by predict.py / run.py.
Prints dead-ball scores on the public window and stoppage stats for the whole match.

    python -m dead_alive_detection.evaluation.bar_window cache/dead_alive/*
"""

import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

# Params
GT_PATH = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
BASELINE = ROOT / "evaluation" / "results" / "baseline_smooth50.csv"
FPS = 25


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def auc(positive, negative):
    a, b = np.asarray(positive, float), np.asarray(negative, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    return (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def labels(path):
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()


def half_of(frame):
    for half, first, last in half_video_blocks(GAME):
        if first <= frame <= last:
            return half
    return None


def load(folder):
    frames = pd.read_csv(Path(folder) / "frames.csv", usecols=["frame", "p_dead", "state"], low_memory=False)
    return {"name": Path(folder).name, "state": frames["state"].to_numpy(),
            "p": frames["p_dead"].to_numpy(dtype=float),
            "stops": pd.read_csv(Path(folder) / "intervals.csv"),
            "submission": labels(Path(folder) / "submission.csv")}


def window_metrics(model, gt):
    (start, stop), = pm.valid_runs(pm.label_codes(gt))
    g, s, p = gt[start:stop], model["state"][start:stop], model["p"][start:stop]
    dead_gt, dead_pred = g == "D", s == "D"
    tp, fp, fn = (dead_gt & dead_pred).sum(), (~dead_gt & dead_pred).sum(), (dead_gt & ~dead_pred).sum()
    pred_runs = runs_of(dead_pred)
    starts, ends, cover = [], [], []
    for a, b in runs_of(dead_gt):
        hits = [(pa, pb) for pa, pb in pred_runs if pb > a and pa < b]
        cover.append(dead_pred[a:b].mean())
        if hits:
            starts.append(abs(hits[0][0] - a) / FPS)
            ends.append(abs(hits[-1][1] - b) / FPS)
    false = [(a, b) for a, b in pred_runs if not dead_gt[a:b].any()]
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(model["submission"]))
    return {
        "D precision": tp / max(1, tp + fp), "D recall": tp / max(1, tp + fn),
        "D F1": 2 * tp / max(1, 2 * tp + fp + fn),
        "p_dead AUC": auc(p[dead_gt], p[~dead_gt]), "p_dead Brier": np.nanmean((p - dead_gt) ** 2),
        "GT D covered (mean)": np.mean(cover), "GT D found": "%d/%d" % (len(starts), len(cover)),
        "|start lag| s (mean)": np.mean(starts) if starts else np.nan,
        "|end lag| s (mean)": np.mean(ends) if ends else np.nan,
        "false stoppages": len(false), "false stoppage s": sum(b - a for a, b in false) / FPS,
        "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "possessions": r["n_pred_spells"],
    }


def match_metrics(model):
    graded = model["state"] != "X"
    stops = model["stops"]
    seconds = stops["seconds"] if len(stops) else pd.Series(dtype=float)
    return {"share Dead": (model["state"][graded] == "D").mean(), "stoppages": len(stops),
            "median stoppage s": seconds.median(), "p90 stoppage s": seconds.quantile(0.9),
            "stoppages < 3 s": int((seconds < 3).sum()), "stoppages > 60 s": int((seconds > 60).sum())}


def smart_tagging(model, xml_path):
    import xml.etree.ElementTree as ET
    hits = {}
    for instance in ET.parse(xml_path).iter("instance"):
        if (instance.findtext("code") or "").strip().lower() != "set pieces":
            continue
        kind = next((label.findtext("text") for label in instance.findall("label")
                     if label.findtext("group") == "Type"), "?")
        a = int(round(float(instance.findtext("start")) * FPS))
        b = int(round(float(instance.findtext("end")) * FPS))
        if half_of(a) is None and half_of(b - 1) is None:
            continue
        hits.setdefault(kind, []).append(bool((model["state"][a:b] == "D").any()))
    row = {"set pieces found": "%d/%d" % (sum(map(sum, hits.values())), sum(map(len, hits.values())))}
    row.update({kind: "%d/%d" % (sum(v), len(v)) for kind, v in sorted(hits.items())})
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--smart-tagging", type=Path)
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    gt = labels(GT_PATH)
    models = [load(f) for f in args.folders if (f / "frames.csv").exists()]

    base = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels(BASELINE)))
    print("baseline_smooth50 on the public window: spellF1 %.4f  macroF1 %.4f  possessions %d (GT 17)\n"
          % (base["spell_f1_mean"], base["macro_f1"], base["n_pred_spells"]))
    print("=== public window ===")
    print(pd.DataFrame({m["name"]: window_metrics(m, gt) for m in models}).round(3).to_string())
    print("\n=== whole match ===")
    print(pd.DataFrame({m["name"]: match_metrics(m) for m in models}).round(3).to_string())
    if len(models) > 1:
        print("\n=== agreement on graded frames (same Dead/alive call) ===")
        graded = models[0]["state"] != "X"
        for a, b in itertools.combinations(models, 2):
            print("  %-12s vs %-12s %.1f%%" % (a["name"], b["name"], 100 * (a["state"][graded] == b["state"][graded]).mean()))
    if args.smart_tagging:
        print("\n=== Smart Tagging set pieces inside a predicted stoppage (weak recall) ===")
        print(pd.DataFrame({m["name"]: smart_tagging(m, args.smart_tagging) for m in models}).to_string())


if __name__ == "__main__":
    main()
