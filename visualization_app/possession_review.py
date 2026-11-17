"""Logic of the Possession review page (no Streamlit): reading predictions, scores, possession changes.

A prediction folder is what possession_model/predict.py writes: frames.csv (frame, p_h, logit_h,
label H/A/D/X) and model.json. Labels are compared with the official metric code
(evaluation/possession_metrics.py) on the public window only; elsewhere there is no ground truth.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402

TEAMS = ("H", "A")


def discover(root):
    """{name: folder} of every prediction folder (a subfolder with frames.csv) under root."""
    root = Path(root)
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.iterdir()) if (p / "frames.csv").exists()}


def load_prediction(folder):
    """(labels (F,) '<U1', p_h (F,) float or None, model.json dict) of a prediction folder."""
    frames = pd.read_csv(Path(folder) / "frames.csv")
    labels = frames["label"].astype(str).str.strip().str.upper().to_numpy().astype("<U1")
    p_h = frames["p_h"].to_numpy(dtype=float) if "p_h" in frames else None
    meta = {}
    if (Path(folder) / "model.json").exists():
        meta = json.loads((Path(folder) / "model.json").read_text(encoding="utf-8"))
    return labels, p_h, meta


def load_labels(path):
    """(F,) '<U1' labels of any frame,label CSV (a submission, a baseline, the GT)."""
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy().astype("<U1")


def spells(labels, start=0, stop=None):
    """[(start, stop, team)] maximal runs of H or of A inside [start, stop) (D and X split them)."""
    stop = len(labels) if stop is None else stop
    part = labels[start:stop]
    if not len(part):
        return []
    edges = np.r_[0, np.flatnonzero(part[1:] != part[:-1]) + 1, len(part)]
    return [(start + int(a), start + int(b), str(part[a])) for a, b in zip(edges[:-1], edges[1:])
            if part[a] in TEAMS]


def window_metrics(pred, gt, start, stop):
    """Official metrics of pred against gt on frames [start, stop)."""
    r = pm.evaluate(pm.label_codes(gt[start:stop]), pm.label_codes(pred[start:stop]))
    t = r["transitions"][25]
    lag = t["mean_abs_latency"]
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "team acc": r["team_acc_gt_alive"],
            "changes F1@1s": t["f1"], "lag (s)": np.nan if lag is None else lag / pm.FPS,
            "possessions": r["n_pred_spells"], "GT possessions": r["n_gt_spells"]}


def match_metrics(labels, fps=25):
    """Whole-match aggregates of one prediction (no ground truth needed)."""
    graded = labels != "X"
    lengths = np.array([b - a for a, b, _ in spells(labels)], dtype=float) / fps
    return {"possessions": len(lengths), "mean s": float(lengths.mean()) if len(lengths) else np.nan,
            "median s": float(np.median(lengths)) if len(lengths) else np.nan,
            "< 1 s": float((lengths < 1).mean()) if len(lengths) else np.nan,
            "H": float((labels[graded] == "H").mean()), "A": float((labels[graded] == "A").mean()),
            "D": float((labels[graded] == "D").mean())}


def possession_starts(labels, start=0, stop=None):
    """DataFrame (frame, team, after): the first frame of every possession in [start, stop), and what
    came before it (the other team, D, or X)."""
    rows = []
    for a, b, team in spells(labels, start, stop):
        before = labels[a - 1] if a > 0 else "X"
        rows.append({"frame": a, "team": team, "after": "other team" if before in TEAMS else before,
                     "seconds": (b - a) / 25})
    return pd.DataFrame(rows, columns=["frame", "team", "after", "seconds"])


def compare_starts(pred, gt, start, stop, tol_frames=25):
    """Every GT possession start in [start, stop) matched to the nearest predicted start of the same
    team within tol_frames (lag = predicted - GT, in s), plus predicted starts no GT start explains.
    DataFrame (frame, team, status, lag_s, after)."""
    gt_starts = possession_starts(gt, start, stop)
    pred_starts = possession_starts(pred, start, stop)
    used, rows = set(), []
    for g in gt_starts.itertuples():
        near = pred_starts[(pred_starts.team == g.team) & ((pred_starts.frame - g.frame).abs() <= tol_frames)
                           & ~pred_starts.index.isin(used)]
        if len(near):
            k = (near.frame - g.frame).abs().idxmin()
            used.add(k)
            rows.append({"frame": int(g.frame), "team": g.team, "status": "found",
                         "lag_s": (int(pred_starts.frame[k]) - int(g.frame)) / 25, "after": g.after})
        else:
            rows.append({"frame": int(g.frame), "team": g.team, "status": "missed", "lag_s": np.nan, "after": g.after})
    for k, s in pred_starts.iterrows():
        if k not in used:
            rows.append({"frame": int(s.frame), "team": s.team, "status": "spurious", "lag_s": np.nan, "after": s.after})
    return pd.DataFrame(rows, columns=["frame", "team", "status", "lag_s", "after"]).sort_values("frame").reset_index(drop=True)


def disagreements(pred, gt, start, stop, min_frames=13):
    """DataFrame (start, stop, seconds, gt, pred): runs inside [start, stop) where the GT is scored
    (H/A/D) and the prediction says something else, at least min_frames long."""
    scored = np.isin(gt[start:stop], ("H", "A", "D"))
    wrong = scored & (pred[start:stop] != gt[start:stop])
    key = np.where(wrong, np.char.add(gt[start:stop], pred[start:stop]), "")
    edges = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1, len(key)]
    rows = [{"start": start + int(a), "stop": start + int(b), "seconds": (b - a) / 25,
             "gt": str(gt[start + a]), "pred": str(pred[start + a])}
            for a, b in zip(edges[:-1], edges[1:]) if key[a] and b - a >= min_frames]
    return pd.DataFrame(rows, columns=["start", "stop", "seconds", "gt", "pred"])


def describe(meta):
    """One line about a prediction folder from its model.json."""
    if not meta:
        return "no model.json"
    dec = meta.get("decoding", {})
    dead = os.path.basename(os.path.dirname(meta.get("dead", ""))) or "?"
    return "run {} · {} model(s) · mask {} · {} T={} w_change={} bandwidth={}".format(
        meta.get("run", "?"), len(meta.get("models", [])), dead, dec.get("method", "HSMM"),
        dec.get("temperature", "?"), dec.get("w_change", "?"), dec.get("bandwidth", "?"))


def runs(labels):
    """DataFrame (start, stop, label): every maximal run of one label (the whole-match timeline)."""
    labels = np.asarray(labels)
    if not len(labels):
        return pd.DataFrame(columns=["start", "stop", "label"])
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return pd.DataFrame({"start": edges[:-1], "stop": edges[1:], "label": labels[edges[:-1]].astype(str)})


def agreement(ref, other):
    """How often two predictions say the same, with no ground truth: on frames both grade (not X),
    on whether the ball is live, and on the team where both say live."""
    graded = (ref != "X") & (other != "X")
    live_r, live_o = np.isin(ref, TEAMS), np.isin(other, TEAMS)
    both = live_r & live_o
    return {"same label": float((ref[graded] == other[graded]).mean()) if graded.any() else np.nan,
            "same live/dead": float((live_r[graded] == live_o[graded]).mean()) if graded.any() else np.nan,
            "same team (both live)": float((ref[both] == other[both]).mean()) if both.any() else np.nan,
            "frames on the other team": int((both & (ref != other)).sum())}


def model_disagreements(labels, min_frames=25, gap=12):
    """DataFrame (start, stop, seconds, kind, <one column per model>): stretches where the models do not
    all say the same thing, closer than `gap` frames merged, at least `min_frames` long. kind is "team"
    when some say H and others A there, else "live/dead". Each model's column is what it says along
    the stretch (runs collapsed, e.g. "H→A")."""
    names = list(labels)
    columns = ["start", "stop", "seconds", "kind"] + names
    if len(names) < 2:
        return pd.DataFrame(columns=columns)
    stack = np.stack([np.asarray(labels[n]) for n in names])
    differ = (stack != stack[0]).any(axis=0)
    edges = np.flatnonzero(np.diff(np.r_[0, differ.astype(np.int8), 0]))
    merged = []
    for a, b in zip(edges[::2], edges[1::2]):
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    rows = []
    for a, b in merged:
        if b - a < min_frames:
            continue
        part = stack[:, a:b]
        team = bool(((part == "H").any(axis=0) & (part == "A").any(axis=0)).any())
        row = {"start": int(a), "stop": int(b), "seconds": (b - a) / 25, "kind": "team" if team else "live/dead"}
        for n, seq in zip(names, part):
            keep = np.r_[True, seq[1:] != seq[:-1]]
            row[n] = "→".join(seq[keep][:6]) + ("…" if keep.sum() > 6 else "")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
