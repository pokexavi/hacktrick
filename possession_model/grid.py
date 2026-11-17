"""
Tries every decoder setting of [decoding.grid] on a finished run's saved outputs, no retraining.
Each fold is scored with the setting that wins on the other folds; writes grid.csv and grid.md.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from possession_model import config, data, decode, evaluate
from possession_model.run import markdown_table, oof_path


def settings(cfg):
    g = cfg["decoding"]["grid"]
    for t, w, bw in itertools.product(g["temperatures"], g["w_change"], g["bandwidths"]):
        yield dict(cfg["decoding"], temperature=t, w_change=w, bandwidth=bw)


def run_grid(run_dir, overrides=None):
    cfg = config.load(os.path.join(run_dir, "config.toml"), overrides)
    name = os.path.basename(os.path.normpath(run_dir))
    cache = os.path.join(config.resolve_path(cfg["runtime"]["cache"]), name)
    _, folds, _ = data.split(cfg)
    rows = []
    for fold in folds:
        path = oof_path(cache, fold, fold, config.PRIMARY)
        if not os.path.exists(path):
            print("no saved outputs for fold {} ({}): skipped".format(fold, path))
            continue
        out = np.load(path)
        prior = decode.Priors.load(np.load(os.path.join(cache, "priors_{}.npz".format(fold))))
        m = data.load(cfg, fold)
        for dec in settings(cfg):
            steps = decode.decode_steps(dec, prior, out["logit"], out["change"], out["live"], out["period"])
            frames = decode.to_frames(steps, out["rows"], m.match.period, out["live_frames"])
            rows.append({"fold": fold, "temperature": dec["temperature"], "w_change": dec["w_change"],
                         "bandwidth": dec["bandwidth"], **evaluate.score(m.reference, frames)})
        print("fold {}: {} settings".format(fold, len(list(settings(cfg)))), flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(run_dir, "grid.csv"), index=False)
    keys = ["temperature", "w_change", "bandwidth"]
    metric = cfg["decoding"]["grid"]["select_metric"]
    chosen = []
    folds_done = grid.fold.unique() if len(grid) else []
    for fold in folds_done if len(folds_done) > 1 else []:
        others = grid[grid.fold != fold].groupby(keys)[metric].mean()
        best = others.idxmax()
        row = grid[(grid.fold == fold) & (grid[keys].apply(tuple, axis=1) == best)].iloc[0]
        chosen.append({"fold": fold, **dict(zip(keys, best)), metric: row[metric]})
    text = ["# Decoder grid: {} ({})".format(name, config.PRIMARY), ""]
    if chosen:
        chosen = pd.DataFrame(chosen)
        text += ["Nested selection (each fold scored with the setting chosen on the others): mean "
                 "{} = {:.4f}".format(metric, chosen[metric].mean()), "", markdown_table(chosen.round(4)), ""]
    else:
        text += ["Nested selection needs at least 2 folds with saved outputs ({} here): only the "
                 "in-sample table below, which is optimistic.".format(len(folds_done)), ""]
    if len(grid):
        overall = grid.groupby(keys)[metric].mean().sort_values(ascending=False).head(10).reset_index()
        text += ["Best settings on all folds (in-sample, for reading only):", "",
                 markdown_table(overall.round(4)), ""]
    with open(os.path.join(run_dir, "grid.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(text))
    print("\n".join(text))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    run_grid(args.run, args.set)


if __name__ == "__main__":
    main()
