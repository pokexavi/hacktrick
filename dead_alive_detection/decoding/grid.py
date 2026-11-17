"""
Scores a run's saved held-out heads with every decoder setting and picks the best one.

    python -m dead_alive_detection.decoding.grid --run <results/run>

No training. Settings are compared with nested leave-one-match-out, so no match chooses its own
setting. Writes grid.csv and report.md.
"""

import argparse
import itertools
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from dead_alive_detection import config as config_lib
from dead_alive_detection.decoding.decoder import labelled, make_decoder
from dead_alive_detection.evaluation import metrics
from dead_alive_detection.evaluation.stoppage_types import fine_stoppages, step_classes
from dead_alive_detection.loading.tensors import match_tensors, to_frames
from external import schema

KEYS = ["temperature", "w_start", "w_stop", "type_weight", "type_evidence"]
UNTYPED = (0.0, "none")          # the type part of an untyped setting
METRICS = ("dead_f1", "d_f1", "start_err_s", "end_err_s", "missed", "false", "n_true", "n_pred",
           "spell_f1", "spell_f1_ctrl", "dead_share_pred")
HIGHER_IS_BETTER = ("dead_f1", "d_f1", "spell_f1", "spell_f1_ctrl")

_cache = {}        # per worker process: npz files, matches, labelled steps


def _npz(path):
    if path not in _cache:
        _cache[path] = dict(np.load(path))
    return _cache[path]


def dfl_match(name):
    key = ("match", name)
    if key not in _cache:
        path = [p for p in schema.processed_paths("dfl") if name in p][0]
        _cache[key] = schema.load_match(path)
    return _cache[key]


def labelled_steps(names):
    """
    Labelled 5 Hz steps of the given DFL matches, cached.
    """
    out = []
    for name in names:
        key = ("labels", name)
        if key not in _cache:
            _cache[key] = labelled([match_tensors(dfl_match(name))])[0]
        out.append(_cache[key])
    return out


def step_types(names, classes):
    """
    Stoppage class of each 5 Hz step of the given DFL matches, -1 outside stoppages.
    """
    out = []
    for name in names:
        key = ("types", name, tuple(classes))
        if key not in _cache:
            match = dfl_match(name)
            truth = metrics.pm.label_codes(metrics.truth_labels(match))
            _cache[key] = step_classes(fine_stoppages(match, truth, classes),
                                       match_tensors(match)["row"], list(classes))
        out.append(_cache[key])
    return out


def decoder_for(train, cfg, setting=None):
    types = step_types(train, cfg["types"]["classes"]) if cfg["decoding"]["typed"]["enabled"] \
        else None
    return make_decoder(labelled_steps(train), types, cfg, setting)


def read_grid(path):
    """
    Reads a grid.csv, adding the type columns if it predates the typed decoder.
    """
    table = pd.read_csv(path)
    for key, value in zip(KEYS[3:], UNTYPED):
        if key not in table:
            table[key] = value
    return table


def oof_path(folder, fold, condition):
    return os.path.join(folder, "oof_{}_{}.npz".format(fold, condition))


def folds_of(folder):
    return sorted({f[len("oof_"):].split("_")[0] for f in os.listdir(folder)
                   if f.startswith("oof_") and f.endswith(".npz")})


def _task(args):
    folder, fold, condition, setting, cfg = args
    d = _npz(oof_path(folder, fold, condition))
    decoder = decoder_for(list(d["train"]), cfg, setting)
    dead = decoder.decode(d["logits"], d["prior"], d["period"])
    match = dfl_match(fold)
    return {"fold": fold, "condition": condition, **dict(zip(KEYS, decoder.setting)),
            **metrics.score(match, to_frames(dead, d["row"], len(match.frame)))}


def settings(cfg):
    """
    Every decoder setting in [decoding.grid], plus its extra ones.
    """
    g = cfg["decoding"]["grid"]
    out = list(itertools.product(g["temperatures"], g["w_start"], g["w_stop"]))
    out += [tuple(s) for s in g["extra"] if tuple(s) not in out]
    if cfg["decoding"]["typed"]["enabled"]:
        return [s + (w, e) for s in out for w in g["type_weights"] for e in g["type_evidence"]]
    return [s + UNTYPED for s in out]


def run_setting(cfg):
    d = cfg["decoding"]
    typed = (d["typed"]["type_weight"], d["typed"]["evidence"]) if d["typed"]["enabled"] \
        else UNTYPED
    return (d["temperature"], d["w_start"], d["w_stop"]) + typed


def grid(folder, cfg, folds=None, table=None, processes=None):
    """
    Scores every fold, condition and setting, skipping rows already in `table`.
    """
    folds = folds or folds_of(folder)
    conditions = [c for c in cfg["augment"]["eval"]["conditions"]
                  if all(os.path.exists(oof_path(folder, f, c)) for f in folds)]
    plan = [(f, c, run_setting(cfg)) for f in folds for c in conditions]
    plan += [(f, c, s) for f in folds for c in cfg["decoding"]["grid"]["conditions"]
             for s in settings(cfg) if c in conditions and (f, c, s) not in plan]
    done = set()
    if table is not None and len(table):
        done = {(r.fold, r.condition, tuple(getattr(r, k) for k in KEYS))
                for r in table.itertuples()}
    tasks = [(folder, f, c, s, cfg) for f, c, s in plan if (f, c, s) not in done]
    if not tasks:
        return table
    with mp.Pool(processes or cfg["runtime"]["processes"]) as pool:
        rows = pool.map(_task, tasks, chunksize=4)
    new = pd.DataFrame(rows)
    return pd.concat([table, new], ignore_index=True) if table is not None and len(table) else new


def nested(table, metric):
    """
    For each condition and held-out fold, the setting chosen on the other folds and its scores.
    """
    rows = []
    for condition, part in table.groupby("condition"):
        if part.groupby(KEYS).ngroups < 2 or part.fold.nunique() < 2:
            continue
        for fold in part.fold.unique():
            others = part[part.fold != fold].groupby(KEYS)[metric].mean()
            chosen = others.idxmax()
            held = part[(part.fold == fold) & (part[KEYS] == list(chosen)).all(axis=1)].iloc[0]
            rows.append({**held.to_dict(), "chosen": chosen})
    return pd.DataFrame(rows)


def at(table, setting):
    return table[(table[KEYS] == list(setting)).all(axis=1)]


def choice(table, cfg):
    """
    Setting with the best mean select_metric over all folds of select_condition.
    """
    g = cfg["decoding"]["grid"]
    part = table[table.condition == g["select_condition"]]
    best = part.groupby(KEYS)[g["select_metric"]].mean().idxmax()
    return tuple(float(v) for v in best[:4]) + (str(best[4]),)


def fmt(df):
    return df.to_string(float_format=lambda v: "{:.3f}".format(v))


def report(table, cfg, title):
    """
    Text of report.md: the run's decoder, nested selection against it and the top settings.
    """
    g = cfg["decoding"]["grid"]
    own = at(table, run_setting(cfg))
    folds = sorted(table.fold.unique())
    lines = ["# Decoder grid: {}".format(title), "",
             "{} folds: {}. {} settings ({}); the run's decoder {} on "
             "every condition, the grid on {}.".format(
                 len(folds), ", ".join(folds), table.groupby(KEYS).ngroups, ", ".join(KEYS), run_setting(cfg),
                 ", ".join(g["conditions"])), "",
             "## 1. The run's decoder, mean over folds", "", "```",
             fmt(own.groupby("condition")[list(METRICS)].mean()), "```", ""]
    chosen = nested(table, g["select_metric"])
    if len(chosen):
        rows = []
        for condition in chosen.condition.unique():
            b = own[own.condition == condition].set_index("fold")
            c = chosen[chosen.condition == condition].set_index("fold")
            for metric in ("dead_f1", "spell_f1", "spell_f1_ctrl", "start_err_s", "end_err_s",
                           "missed", "false"):
                delta = c[metric] - b[metric].reindex(c.index)
                better = delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
                rows.append({"condition": condition, "metric": metric, "run decoder": b[metric].mean(),
                             "selected": c[metric].mean(), "delta": delta.mean(),
                             "folds better": "{}/{}".format(int(better.sum()), len(delta))})
        lines += ["## 2. Nested leave-one-out selection (on {}) vs the run's decoder".format(
                      g["select_metric"]), "", "```",
                  fmt(pd.DataFrame(rows).set_index(["condition", "metric"])), "```", "",
                  "Settings chosen per fold ({}):".format(", ".join(KEYS)), "", "```",
                  fmt(chosen.pivot(index="fold", columns="condition", values="chosen")), "```", ""]
    if g["select_condition"] in set(table.condition):
        sel = table[table.condition == g["select_condition"]].groupby(KEYS)[list(METRICS)].mean()
        lines += ["## 3. Top settings on all folds ({})".format(g["select_condition"]), "", "```",
                  fmt(sel.sort_values(g["select_metric"], ascending=False).head(8)), "```", "",
                  "Grid's choice ({}): {}".format(", ".join(KEYS), choice(table, cfg)),
                  ""]
    return "\n".join(lines)


def run_grid(folder, cfg, out, title=None, folds=None):
    """
    Writes grid.csv and report.md to out from the heads in folder.
    Returns the table and the report text.
    """
    path = os.path.join(out, "grid.csv")
    table = read_grid(path) if os.path.exists(path) else None
    table = grid(folder, cfg, folds, table)
    table.to_csv(path, index=False, float_format="%.5f")
    text = report(table, cfg, title or folder)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(text)
    return table, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="results/<run> folder: its config and saved heads")
    parser.add_argument("--oof", help="folder of saved heads (default: the run's)")
    parser.add_argument("--config", help="config (default: the run's)")
    parser.add_argument("--out", help="where grid.csv and report.md go (default: the run folder)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. decoding.grid.extra=[[0.1,3.0,0.0]]")
    args = parser.parse_args(argv)
    if not args.run and not (args.oof and args.config and args.out):
        parser.error("give --run, or --oof, --config and --out")
    cfg = config_lib.load(args.config or os.path.join(args.run, "config.toml"), args.set)
    folder = args.oof or os.path.join(config_lib.resolve_path(cfg["runtime"]["cache"]),
                                      os.path.basename(os.path.normpath(args.run)))
    out = args.out or args.run
    os.makedirs(out, exist_ok=True)
    print(run_grid(folder, cfg, out)[1])


if __name__ == "__main__":
    main()
