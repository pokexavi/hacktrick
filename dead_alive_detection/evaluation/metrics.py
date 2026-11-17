"""
Scores a predicted dead-ball mask against DFL's labels at 25 fps.

The main score is dead_f1, a segment F1 over dead stretches averaged over the spellF1 IoU
thresholds. Half-edge artefacts in the labels are set to X first (loading/clean.py).
"""

import os
import sys

from dataclasses import replace

import numpy as np

from dead_alive_detection.loading.clean import mask_edge_artifacts
from external import labels as dfl_labels

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
CONTROLLED_FRAMES = 50
MATCH_IOU = 0.3


def scored_runs(gt_codes, period):
    """
    Labelled (start, stop) spans, also split where the period changes.
    Without that split, a stoppage ending one DFL half would merge with one opening the next.
    """
    key = np.where(gt_codes == pm.X, 0, period)
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1, [len(key)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if key[a] > 0]


def dead_segments(dead, runs):
    """
    Maximal dead stretches of a boolean mask, never crossing a run.
    """
    codes = np.where(dead, pm.D, pm.H)
    return [s for s in pm.segments(codes, runs) if s[2] == pm.D]


def dead_f1(gt_codes, pred_dead, period, ious=pm.SPELL_IOUS):
    """
    F1 of dead-ball segments at each IoU threshold, and their mean.
    Frames labelled X are not scored.
    """
    runs = scored_runs(gt_codes, period)
    gt = dead_segments(gt_codes == pm.D, runs)
    pred = dead_segments(pred_dead, runs)
    overlaps = pm.spell_overlaps(gt, pred)
    f1 = {}
    for iou in ious:
        tp, fp, fn = pm.match_spells(overlaps, len(gt), len(pred), iou)
        f1[iou] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"f1": f1, "mean_f1": float(np.mean(list(f1.values()))),
            "n_gt": len(gt), "n_pred": len(pred)}


def truth_labels(match):
    return mask_edge_artifacts(match.label, match.period)


def _restart_team(label, period):
    """
    Per frame, the team of the next live frame in the same half, or of the previous one if none.
    """
    team = np.where(np.isin(label, ("H", "A")), label, "")
    out = team.copy()
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        if not len(rows):
            continue
        part = team[rows]
        # next non-empty, filled backwards; then previous non-empty for a trailing stretch
        idx = np.where(part != "", np.arange(len(part)), len(part))
        nxt = np.minimum.accumulate(idx[::-1])[::-1]
        filled = np.where(nxt < len(part), part[np.minimum(nxt, len(part) - 1)], "")
        prv = np.maximum.accumulate(np.where(part != "", np.arange(len(part)), -1))
        filled = np.where(filled != "", filled, np.where(prv >= 0, part[np.maximum(prv, 0)], ""))
        out[rows] = filled
    return out


def _with_true_teams(label, period, pred_dead):
    """
    Submission-like labels: D where predicted dead, otherwise the true or restarting team.
    """
    team = _restart_team(label, period)
    out = np.where(pred_dead, "D", np.where(team != "", team, "X"))
    return np.where(label == "X", "X", out)


def _spell_f1(label, period, pred_dead):
    gt = pm.label_codes(label)
    pred = pm.label_codes(_with_true_teams(label, period, pred_dead))
    runs = scored_runs(gt, period)
    spells = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return spells["mean_f1"], spells["n_gt"], spells["n_pred"]


def boundary_errors(gt_segments, pred_segments, fps=25):
    """
    Start and end errors in seconds of the true stoppages matched at MATCH_IOU,
    plus the number of missed and false stoppages.
    """
    overlaps = pm.spell_overlaps(gt_segments, pred_segments)
    pairs, used_gt, used_pred = [], set(), set()
    for iou, g, p in overlaps:                         # IoU-descending, as the official matcher
        if iou < MATCH_IOU:
            break
        if g in used_gt or p in used_pred:
            continue
        pairs.append((g, p))
        used_gt.add(g)
        used_pred.add(p)
    start = np.array([abs(gt_segments[g][0] - pred_segments[p][0]) for g, p in pairs]) / fps
    end = np.array([abs(gt_segments[g][1] - pred_segments[p][1]) for g, p in pairs]) / fps
    return start, end, len(gt_segments) - len(pairs), len(pred_segments) - len(pairs)


def score(match, pred_dead):
    """
    All metrics of a 25 fps predicted dead mask on a DFL match, as a dict.
    """
    label = truth_labels(match)
    period = match.period
    gt = pm.label_codes(label)
    pred_dead = np.asarray(pred_dead, dtype=bool) & (label != "X")
    runs = scored_runs(gt, period)
    gt_seg = dead_segments(gt == pm.D, runs)
    pred_seg = dead_segments(pred_dead, runs)

    scored = label != "X"
    true_dead = label == "D"
    tp = int((pred_dead & true_dead).sum())
    precision = tp / max(int((pred_dead & scored).sum()), 1)
    recall = tp / max(int(true_dead.sum()), 1)
    start, end, missed, false = boundary_errors(gt_seg, pred_seg)
    spell, n_poss, n_pred_poss = _spell_f1(label, period, pred_dead)
    ctrl = dfl_labels.controlled(label, period, CONTROLLED_FRAMES)
    spell_ctrl, _, _ = _spell_f1(ctrl, period, pred_dead)
    segment_f1 = dead_f1(gt, pred_dead, period)
    return {"dead_f1": segment_f1["mean_f1"],
            **{"f1@{}".format(iou): f for iou, f in segment_f1["f1"].items()},
            "d_precision": precision, "d_recall": recall,
            "d_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "start_err_s": float(start.mean()) if len(start) else np.nan,
            "end_err_s": float(end.mean()) if len(end) else np.nan,
            "n_true": len(gt_seg), "n_pred": len(pred_seg), "missed": missed, "false": false,
            "spell_f1": spell, "spell_f1_ctrl": spell_ctrl,
            "dead_share_true": float(true_dead[scored].mean()),
            "dead_share_pred": float(pred_dead[scored].mean())}


def perfect(match):
    """
    Scores of the true dead mask, the best a model can get.
    """
    return score(match, truth_labels(match) == "D")


def with_truth(match):
    """
    The match with the cleaned truth labels used for scoring.
    """
    return replace(match, label=truth_labels(match))
