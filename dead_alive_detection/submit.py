"""
Build a complete submission: D from a dead-ball model folder, H/A from any team source.

    python -m dead_alive_detection.submit --dead cache/dead_alive/tcn_bar_boundary --teams evaluation/results/baseline_smooth50.csv --out outputs/submissions/tcn_bar_smooth50.csv

The file is checked with the official validator. It also writes <out>_dead.csv, the dead-ball mask
alone.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from external import schema

ROOT = schema.ROOT
sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

# Params
SECOND_HALF_KICKOFF = 113742       # video frame (FAQ.md); graded from 110,808, all D before this
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")


def read_labels(path):
    """
    Upper-case labels of a frame,label CSV.
    """
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()


def build(frames, teams):
    """
    Labels for every video frame and the dead-ball mask (bool).
    """
    n = len(teams)
    if len(frames) != n or not np.array_equal(frames["frame"].to_numpy(), np.arange(n)):
        raise ValueError("model frames and team labels must both cover video frames 0..{}".format(n - 1))
    state = frames["state"].to_numpy()
    labels = np.full(n, "X", dtype="<U1")
    dead = np.zeros(n, dtype=bool)
    for half, first, last in half_video_blocks(GAME):
        rows = np.arange(first, last + 1)
        d = state[rows] == "D"
        if half == 2:
            d |= rows < SECOND_HALF_KICKOFF
        team = pd.Series(np.where(np.isin(teams[rows], ("H", "A")), teams[rows], None))
        team = team.ffill().bfill().fillna("X").to_numpy()
        labels[rows] = np.where(d, "D", team)
        dead[rows] = d
    return labels, dead


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", required=True, help="cache/dead_alive/<model> folder")
    parser.add_argument("--teams", default=os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"))
    parser.add_argument("--out", required=True, help="submission CSV to write (keep it out of git)")
    args = parser.parse_args(argv)

    frames = pd.read_csv(os.path.join(args.dead, "frames.csv"), usecols=["frame", "p_dead", "state"])
    labels, dead = build(frames, read_labels(args.teams))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(args.out, index=False)
    mask = np.where(labels == "X", "X", np.where(dead, "D", "L"))
    pd.DataFrame({"frame": np.arange(len(labels)), "label": mask, "p_dead": frames["p_dead"]}).to_csv(
        os.path.splitext(args.out)[0] + "_dead.csv", index=False, float_format="%.4g")

    graded = labels != "X"
    print("{}: {} graded frames, D {:.1%}, H {:.1%}, A {:.1%}".format(
        args.out, graded.sum(), *(np.mean(labels[graded] == c) for c in "DHA")))
    r = pm.evaluate(pm.label_codes(read_labels(GT_PUBLIC)), pm.label_codes(labels))
    print("public window (pipeline check only): spellF1 {:.4f}  macroF1 {:.4f}  possessions {}".format(
        r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
    sys.exit(subprocess.call([sys.executable, VALIDATOR, args.out]))


if __name__ == "__main__":
    main()
