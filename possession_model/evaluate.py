"""
Scores a predicted match against its labels with the official metric code.
spell_f1_mean is the challenge's metric; the rest help to understand it.
"""

import numpy as np

from evaluation import possession_metrics as pm

KEYS = ("spell_f1_mean", "macro_f1", "team_acc", "trans_f1_05s", "trans_f1_1s", "lag_1s",
        "n_gt", "n_pred")


def score(reference, predicted):
    gt, pred = pm.label_codes(reference), pm.label_codes(predicted)
    r = pm.evaluate(gt, pred)
    t05, t1 = r["transitions"][12], r["transitions"][25]      # changes found within 0.5 s and 1 s
    lag = t1["mean_abs_latency"]
    return {"spell_f1_mean": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
            "team_acc": r["team_acc_gt_alive"], "trans_f1_05s": t05["f1"], "trans_f1_1s": t1["f1"],
            "lag_1s": np.nan if lag is None else lag / pm.FPS,
            "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"]}
