"""
Score the dead-ball model on the DFL test match, clean and corrupted like BAR-ATH.

    python -m dead_alive_detection.analysis.test_match

One model is trained on corrupted DFL (--train final) and one on clean DFL (--train clean); the test
match is never seen in training. Writes results/test_match/.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from dead_alive_detection import config as config_lib
from dead_alive_detection.evaluation import metrics
from dead_alive_detection.evaluation.metrics import pm
from dead_alive_detection.loading.tensors import to_frames

CONFIG = "dead_alive_detection/configs/tcn_bar_keeperhold.toml"
OUT = "dead_alive_detection/results/test_match"
THREADS = 6                     # two models train side by side on a 12-core CPU
CORRUPTED = {"players": "learned", "dead_ball": "bar", "ball": "hidden", "whistle": "bar",
             "keeper": "hide"}


def model_config(model):
    cfg = config_lib.load(config_lib.resolve_path(CONFIG))
    cfg["name"] = "test_match_" + model
    cfg["runtime"]["threads"] = THREADS
    cfg["augment"]["recipes"]["corrupted"] = dict(CORRUPTED)
    cfg["augment"]["eval"]["conditions"] = ["clean", "corrupted"]
    cfg["augment"]["eval"]["primary"] = "corrupted"
    if model == "clean":
        cfg["augment"]["train"]["recipe"] = "clean"
    return cfg


def model_path(cfg):
    cache = config_lib.resolve_path(cfg["runtime"]["cache"])
    return os.path.join(cache, cfg["name"], "models", cfg["data"]["test_match"] + ".pt")


def macro_f1(match, pred_dead):
    """
    Official macro F1 over H / A / D, using DFL's true H/A so only D varies.
    """
    label = metrics.truth_labels(match)
    pred_dead = np.asarray(pred_dead, bool) & (label != "X")
    gt = pm.label_codes(label)
    pred = pm.label_codes(metrics._with_true_teams(label, match.period, pred_dead))
    runs = metrics.scored_runs(gt, match.period)
    cm = pm.scored_confusion(pm.confusion(gt, pm.fill_abstentions(pred, runs)))
    return float(pm.macro_f1(cm))


def main(argv=None):
    from dead_alive_detection.run import Data, load_or_train
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", choices=["final", "clean"], help="train one model and stop")
    args = parser.parse_args(argv)
    say = lambda text: print(text, flush=True)          # noqa: E731
    models = [args.train] if args.train else ["final", "clean"]
    rows = []
    os.makedirs(config_lib.resolve_path(OUT), exist_ok=True)
    for model in models:
        cfg = model_config(model)
        torch.set_num_threads(cfg["runtime"]["threads"])
        config_lib.dump(cfg, os.path.join(config_lib.resolve_path(OUT), "config_{}.toml".format(model)))
        data = Data(cfg)
        test = cfg["data"]["test_match"]
        dev = [name for name in data.matches if name != test]
        path = model_path(cfg)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trained, prior = load_or_train(data, dev, cfg, path, say)
        if args.train:
            return
        decoder = data.decoder(dev)
        match = data.matches[test]
        for condition, (t, x) in data.conditions(test, ["clean", "corrupted"]).items():
            logits = trained.heads(x, t)
            p = 1 / (1 + np.exp(-logits[:, 0]))
            for name, dead in (("HSMM", decoder.decode(logits, prior, t["period"])),
                               ("threshold 0.5", p > 0.5)):
                pred = to_frames(dead, t["row"], len(match.frame))
                s = metrics.score(match, pred)
                rows.append({"model": model, "decoder": name, "condition": condition,
                             "spell_f1": s["spell_f1"], "macro_f1": macro_f1(match, pred),
                             "dead_f1": s["dead_f1"], "n_pred": s["n_pred"], "n_true": s["n_true"],
                             "missed": s["missed"], "false": s["false"],
                             "start_err_s": s["start_err_s"], "end_err_s": s["end_err_s"],
                             "ball_hidden": float(1 - t["ball_seen"][t["scored"]].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(config_lib.resolve_path(OUT), "test_match.csv"), index=False,
                 float_format="%.4f")
    print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
