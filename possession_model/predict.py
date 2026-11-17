"""
Predicts H / A on BAR with a trained run, inside the live stretches of the dead-ball model.
Averages all the run's models and writes frames.csv, labels.csv and model.json (never commit them).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from external import schema
from possession_model import config, decode, features, train


def load_models(cfg, cache, which="folds"):
    # "folds": the 6 fold models (x seeds). "test": the ones trained on all 6 dev matches (run --test).
    # The duration priors of all of them are pooled.
    tags = sorted(os.path.basename(p)[len("priors_"):-len(".npz")]
                  for p in glob.glob(os.path.join(cache, "priors_*.npz")))
    tags = ["test"] if which == "test" else [t for t in tags if t != "test"]
    files = {t: sorted(glob.glob(os.path.join(cache, "model_{}_s*.pt".format(t)))) for t in tags}
    tags = [t for t in tags if files[t]]
    if not tags:
        raise FileNotFoundError("no {} models in {}".format(which, cache))
    loaded = [decode.Priors.load(np.load(os.path.join(cache, "priors_{}.npz".format(t)))) for t in tags]
    priors = decode.Priors(np.concatenate([p.lengths for p in loaded]), float(np.mean([p.p_h for p in loaded])),
                           float(np.mean([p.p_change for p in loaded])), loaded[0].hz)
    models, names = [], []
    for t in tags:
        for path in files[t]:
            model = train.PossessionModel(cfg)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            models.append(model)
            names.append(os.path.basename(path))
    return names, priors, models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run", required=True, help="possession_model/results/<run>")
    parser.add_argument("--match", required=True, help="a schema parquet (external/convert_atd.py)")
    parser.add_argument("--dead", required=True,
                        help="the dead-ball model's frames.csv (its state column: D, X or live)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", choices=("folds", "test"), default="folds",
                        help="folds: every fold model; test: the models of run --test (all 6 dev matches)")
    parser.add_argument("--set", action="append", default=[], help="e.g. decoding.temperature=0.25")
    args = parser.parse_args(argv)

    # The run may come from Kaggle: here there is no bundle and everything runs on CPU
    cfg = config.load(os.path.join(args.run, "config.toml"),
                      ["runtime.bundle=''", "runtime.device='cpu'"] + args.set)
    torch.set_num_threads(cfg["runtime"]["threads"])
    name = os.path.basename(os.path.normpath(args.run))
    tags, priors, models = load_models(cfg, os.path.join(config.resolve_path(cfg["runtime"]["cache"]), name),
                                       args.models)

    match = schema.load_match(args.match)
    frames = pd.read_csv(args.dead)
    if len(frames) != len(match.frame) or not np.array_equal(frames["frame"].to_numpy(), match.frame):
        raise ValueError("--dead frames do not match the match's frames")
    live = ~np.isin(frames["state"].astype(str).to_numpy(), ("D", "X")) & (match.period > 0)

    it = features.build(cfg, match, live)
    outs = [train.predict(cfg, model, it) for model in models]
    logit = np.mean([o[0] for o in outs], axis=0)
    change = np.mean([o[1] for o in outs], axis=0)
    steps = decode.decode_steps(cfg["decoding"], priors, logit, change, it["live"], it["period"])
    labels = decode.to_frames(steps, it["rows"], match.period, it["live_frames"])

    os.makedirs(args.out, exist_ok=True)
    nearest = np.clip(np.searchsorted(it["rows"], np.arange(len(match.frame))), 0, len(it["rows"]) - 1)
    frame_logit = logit[nearest]
    pd.DataFrame({"frame": match.frame, "p_h": np.round(1 / (1 + np.exp(-frame_logit)), 4),
                  "logit_h": np.round(frame_logit, 3), "label": labels}).to_csv(
        os.path.join(args.out, "frames.csv"), index=False)
    pd.DataFrame({"frame": match.frame, "label": labels}).to_csv(os.path.join(args.out, "labels.csv"), index=False)
    with open(os.path.join(args.out, "model.json"), "w", encoding="utf-8") as handle:
        json.dump({"run": name, "models": tags, "dead": os.path.abspath(args.dead),
                   "decoding": {k: v for k, v in cfg["decoding"].items() if k != "grid"}}, handle, indent=2)
    spells = [s for s in schema.spells(labels, match.period) if s[2] in ("H", "A")]
    graded = labels != "X"
    print("{} ({} models): {} possessions, H {:.1%}, A {:.1%}, D {:.1%} of graded frames -> {}".format(
        name, len(tags), len(spells), np.mean(labels[graded] == "H"), np.mean(labels[graded] == "A"),
        np.mean(labels[graded] == "D"), args.out))


if __name__ == "__main__":
    main()
