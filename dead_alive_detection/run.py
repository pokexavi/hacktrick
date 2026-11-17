"""
Runs one dead-ball experiment from its config: train and score on DFL, then predict BAR-ATH.

    python -m dead_alive_detection.run --config <toml>

Each development match is scored by a model trained on the others, and the test match is scored
once at the end with --test. Finished stages are skipped, so --resume continues a run.
Results go to results/<run>/, logs to logs/<run>/ and models and heads to the cache.
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from dead_alive_detection import config as config_lib
from dead_alive_detection.augment.views import Views, inputs
from dead_alive_detection.decoding.decoder import labelled, logit, make_decoder, type_probabilities
from dead_alive_detection.evaluation import metrics
from dead_alive_detection.evaluation.stoppage_types import (fine_stoppages, recall_by_group,
                                                            step_classes, true_stoppages)
from dead_alive_detection.loading.clean import played
from dead_alive_detection.loading.tensors import match_tensors, to_frames
from external import schema

PACKAGE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PACKAGE, "results")
LOGS = os.path.join(PACKAGE, "logs")


def load_matches(datasets):
    """
    Loads the matches of these sources, with each one's index in the full list.
    The index seeds the corruption draws, so loading fewer sources does not change them.
    """
    matches, index = {}, {}
    for i, path in enumerate(schema.processed_paths("dfl")):
        match = schema.load_match(path)
        if match.source in datasets:
            matches[match.name], index[match.name] = match, i
    return matches, index


class Data:
    """
    The matches with their labels and views, built lazily and cached.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.views = Views(cfg)
        self.matches, self.index = load_matches(cfg["data"]["datasets"])
        self.classes = list(cfg["types"]["classes"])
        self._train, self._labels, self._stoppages, self._types = {}, {}, {}, {}

    def step_types(self, name):
        """
        Stoppage class of each 5 Hz step of a match, -1 outside stoppages.
        """
        if name not in self._types:
            match = self.matches[name]
            truth = metrics.pm.label_codes(metrics.truth_labels(match))
            stops = fine_stoppages(match, truth, self.classes)
            if stops is None:
                raise FileNotFoundError("no DFL events for {}: needed by [types]".format(name))
            self._types[name] = step_classes(stops, match_tensors(match)["row"], self.classes)
        return self._types[name]

    def _with_types(self, name, views):
        """
        Adds the step types to the views when the config has [types] classes.
        """
        if not self.classes:
            return views
        kind = self.step_types(name)
        for t, _ in views:
            t["type"] = kind
        return views

    def training(self, name):
        if name not in self._train:
            self._train[name] = self._with_types(
                name, self.views.training(self.matches[name], self.index[name]))
        return self._train[name]

    def conditions(self, name, names=None):
        views = self.views.conditions(self.matches[name], self.index[name], names)
        self._with_types(name, views.values())
        return views

    def labelled(self, names):
        for name in names:
            if name not in self._labels:
                self._labels[name] = labelled([match_tensors(self.matches[name])])[0]
        return [self._labels[name] for name in names]

    def stoppages(self, name):
        """
        True stoppages of a match with their kind, from the DFL events.
        """
        if name not in self._stoppages:
            match = self.matches[name]
            truth = metrics.pm.label_codes(metrics.truth_labels(match))
            self._stoppages[name] = true_stoppages(match, truth)
        return self._stoppages[name]

    def decoder(self, names, setting=None):
        types = [self.step_types(n) for n in names] if self.cfg["decoding"]["typed"]["enabled"] \
            else None
        return make_decoder(self.labelled(names), types, self.cfg, setting)


def score(match, t_rows, dead_steps, stoppages):
    """
    Scores a 5 Hz dead mask at 25 fps, plus the share of true stoppages of each kind it finds.
    """
    pred = to_frames(dead_steps, t_rows, len(match.frame))
    truth = metrics.pm.label_codes(metrics.truth_labels(match))
    found = recall_by_group(stoppages, pred, truth, match.period) if stoppages else {}
    return {**metrics.score(match, pred),
            **{"found_" + kind.replace(" ", "_"): f for kind, f in found.items()}}


def type_scores(logits, kind, classes):
    """
    Accuracy of the type output on a held-out match's true stoppages, per step, per stoppage
    and per class.
    """
    if not classes or logits.shape[1] <= 3:
        return {}
    p = type_probabilities(logits)
    steps = kind >= 0
    out = {"type_acc_steps": float((np.argmax(p[steps], axis=1) == kind[steps]).mean())}
    change = np.flatnonzero(kind[1:] != kind[:-1]) + 1
    hits = {k: [] for k in range(len(classes))}
    for a, b in zip(np.r_[0, change], np.r_[change, len(kind)]):
        if kind[a] >= 0:
            hits[int(kind[a])].append(int(np.argmax(np.nanmean(p[a:b], axis=0))) == kind[a])
    every = [h for v in hits.values() for h in v]
    out["type_acc_stoppages"] = float(np.mean(every)) if every else np.nan
    out.update({"type_recall_" + c: float(np.mean(hits[k])) if hits[k] else np.nan
                for k, c in enumerate(classes)})
    return out


def train(data, names, cfg, model_path, monitor=None):
    """
    Trains the TCN on the given matches and saves it to model_path.
    Returns the model and the training log-odds of dead.
    """
    from dead_alive_detection.training.fit_tcn import fit_tcn     # torch training only when needed
    views = [data.training(n) for n in names]
    y = np.concatenate([t["dead"][t["scored"]] for match in views for t, _ in match])
    prior = float(logit(y.mean()))
    trained = fit_tcn(views, cfg, monitor)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trained.save(model_path, prior=prior, train=list(names), features=cfg["features"],
                 classes=list(cfg["types"]["classes"]))
    return trained, prior


def load_or_train(data, names, cfg, model_path, say, monitor=None):
    from dead_alive_detection.training.fit_tcn import Trained
    if os.path.exists(model_path):
        trained, extra = Trained.load(model_path)
        if list(extra["train"]) != list(names):
            raise ValueError("{} was trained on {}, not {}".format(model_path, extra["train"], names))
        return trained, extra["prior"]
    started = time.time()
    trained, prior = train(data, names, cfg, model_path, monitor)
    say("trained on {} matches in {:.1f} min".format(len(names), (time.time() - started) / 60))
    return trained, prior


def save_heads(path, t, logits, prior, names):
    extra = {"type": t["type"]} if "type" in t else {}
    np.savez_compressed(path, row=t["row"], period=t["period"], dead=t["dead"], scored=t["scored"],
                        ball_seen=t["ball_seen"], logits=logits, prior=prior, train=np.array(names),
                        **extra)


def oof_path(folder, held, condition):
    return os.path.join(folder, "oof_{}_{}.npz".format(held, condition))


def run_folds(data, folds, dev, cfg, cache, say, log_epoch):
    """
    Trains every fold whose heads are missing and saves its heads per condition.
    """
    conditions = cfg["augment"]["eval"]["conditions"]
    primary = cfg["augment"]["eval"]["primary"]
    for held in folds:
        missing = [c for c in conditions if not os.path.exists(oof_path(cache, held, c))]
        if not missing:
            continue
        names = [n for n in dev if n != held]
        model_path = os.path.join(cache, "models", held + ".pt")
        views = data.conditions(held, sorted(set(missing) | ({primary} if log_epoch else set())))
        monitor = None
        if log_epoch and not os.path.exists(model_path):
            decoder, (t, x) = data.decoder(names), views[primary]
            stoppages = data.stoppages(held)
            prior = float(logit(np.concatenate(
                [v["dead"][v["scored"]] for n in names for v, _ in data.training(n)]).mean()))

            def monitor(epoch, losses, trained):
                logits = trained.heads(x, t)
                p = np.clip(1 / (1 + np.exp(-logits[t["scored"], 0].astype(float))), 1e-6, 1 - 1e-6)
                dead = t["dead"][t["scored"]]
                val = float(-np.mean(dead * np.log(p) + (1 - dead) * np.log(1 - p)))
                result = score(data.matches[held], t["row"], decoder.decode(logits, prior, t["period"]),
                               stoppages)
                log_epoch({"fold": held, "epoch": epoch, **losses, "val_dead_loss": val,
                           "dead_f1": result["dead_f1"]})
        say("fold {}: {} conditions to save".format(held, len(missing)))
        trained, prior = load_or_train(data, names, cfg, model_path, say, monitor)
        for condition in missing:
            t, x = views[condition]
            save_heads(oof_path(cache, held, condition), t, trained.heads(x, t), prior, names)
        say("fold {}: saved {}".format(held, ", ".join(missing)))


def score_folds(data, folds, cfg, cache):
    """
    Decodes and scores every saved fold and condition, one row each.
    """
    rows = []
    for held in folds:
        for condition in cfg["augment"]["eval"]["conditions"]:
            path = oof_path(cache, held, condition)
            if not os.path.exists(path):
                continue
            d = dict(np.load(path))
            dead = data.decoder(list(d["train"])).decode(d["logits"], d["prior"], d["period"])
            types = type_scores(d["logits"], d["type"], data.classes) if "type" in d else {}
            rows.append({"match": held, "condition": condition,
                         **score(data.matches[held], d["row"], dead, data.stoppages(held)),
                         **types})
    return pd.DataFrame(rows)


def with_means(table):
    means = table.drop(columns="match").groupby("condition", sort=False).mean().reset_index()
    means.insert(0, "match", "mean")
    return pd.concat([table, means], ignore_index=True)


def summary(table, cfg):
    """
    Mean headline scores over folds per condition, primary condition first.
    """
    cols = ["dead_f1", "spell_f1", "spell_f1_ctrl", "d_f1", "start_err_s", "end_err_s",
            "missed", "false"] + [c for c in ("type_acc_steps", "type_acc_stoppages") if c in table]
    mean = table[table.match != "mean"].groupby("condition")[cols].mean()
    std = table[table.match != "mean"].groupby("condition")["dead_f1"].std()
    mean.insert(1, "dead_f1_std", std)
    primary = cfg["augment"]["eval"]["primary"]
    return mean.loc[[primary] + [c for c in mean.index if c != primary]]


def new_run_name(name):
    os.makedirs(RESULTS, exist_ok=True)
    taken = [int(n.rsplit("_", 1)[1]) for n in os.listdir(RESULTS)
             if re.fullmatch(re.escape(name) + r"_\d+", n)]
    return "{}_{}".format(name, max(taken, default=0) + 1)


def logger(folder):
    def say(line):
        line = time.strftime("%H:%M:%S ") + line
        print(line, flush=True)
        if folder is not None:
            with open(os.path.join(folder, "train.log"), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    return say


def run_bar(data, dev, cfg, cache, out, say):
    """
    Predicts BAR-ATH with a model trained on the DFL matches and writes its folder.
    """
    from dead_alive_detection.decoding import grid
    from dead_alive_detection.predict import DEAD_ALIVE, write_folder
    path = os.path.join(cache, "bar.npz")
    match = played(schema.load_match(config_lib.resolve_path(cfg["bar"]["match"])))
    if not os.path.exists(path):
        # [bar] train = "all": the test match too (every DFL match of the run's datasets): a better
        # trained BAR model, after the test match has been scored (run --test) or given up.
        if cfg["bar"]["train"] not in ("dev", "all"):
            raise ValueError("[bar] train must be dev or all, not {}".format(cfg["bar"]["train"]))
        names = dev + [cfg["data"]["test_match"]] if cfg["bar"]["train"] == "all" else dev
        if names != dev:
            say("BAR model trained on all {} matches, the test match {} included".format(
                len(names), cfg["data"]["test_match"]))
        trained, prior = load_or_train(data, names, cfg, os.path.join(cache, "models", "BAR-ATH.pt"), say)
        t = match_tensors(match)                   # BAR's ball as it is: it is already lost
        if cfg["features"]["whistle"]:             # the detector's whistles on the broadcast audio
            from dead_alive_detection.augment.whistle import simulated
            table = pd.read_csv(config_lib.resolve_path(cfg["bar"]["whistles"]))
            t["whistle"] = simulated.to_steps(simulated.detected(table, match.frame), t["row"])
            say("BAR whistles: {} from {} ({} steps flagged)".format(
                len(table), cfg["bar"]["whistles"], int(t["whistle"].sum())))
        save_heads(path, t, trained.heads(inputs(t, cfg), t), prior, dev)
    bar = dict(np.load(path))
    setting = None
    if cfg["bar"]["decoder"] == "grid":
        setting = grid.choice(grid.read_grid(os.path.join(out, "grid.csv")), cfg)
    decoder = data.decoder(list(bar["train"]), setting)
    name = cfg["bar"]["name"] or os.path.basename(out)
    write_folder(os.path.join(DEAD_ALIVE, name), match, bar, decoder, cfg,
                 {"run": os.path.relpath(out, schema.ROOT).replace("\\", "/")})


def run_test(data, test, dev, cfg, cache, out, say):
    names = cfg["augment"]["eval"]["conditions"]
    trained, prior = load_or_train(data, dev, cfg, os.path.join(cache, "models", test + ".pt"), say)
    decoder = data.decoder(dev)
    rows = []
    for condition, (t, x) in data.conditions(test, names).items():
        dead = decoder.decode(trained.heads(x, t), prior, t["period"])
        rows.append({"match": test, "condition": condition,
                     **score(data.matches[test], t["row"], dead, data.stoppages(test))})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "test.csv"), index=False, float_format="%.4f")
    say("test match {}:\n{}".format(test, table.set_index("condition").drop(columns="match")[
        ["dead_f1", "spell_f1", "spell_f1_ctrl", "start_err_s", "end_err_s"]].to_string(
        float_format="{:.3f}".format)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="config of a new run")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value (new runs only), e.g. train.epochs=1")
    parser.add_argument("--resume", metavar="RUN_DIR", help="results/<run>: continue it")
    parser.add_argument("--test", metavar="RUN_DIR", help="results/<run>: score the test match once")
    parser.add_argument("--bar-only", action="store_true",
                        help="skip the folds, scores and grid: train the BAR model and write its folder")
    args = parser.parse_args(argv)
    if sum(map(bool, (args.config, args.resume, args.test))) != 1:
        parser.error("give exactly one of --config, --resume, --test")
    if args.set and not args.config:
        parser.error("--set changes a new run only: a saved run keeps its config")
    if args.bar_only and args.test:
        parser.error("--bar-only trains the BAR model: it cannot go with --test")

    if args.config:
        cfg = config_lib.load(args.config, args.set)
        out = os.path.join(RESULTS, new_run_name(cfg["name"]))
        os.makedirs(out)
        config_lib.dump(cfg, os.path.join(out, "config.toml"))
    else:
        out = os.path.normpath(args.resume or args.test)
        cfg = config_lib.load(os.path.join(out, "config.toml"))
        if args.test and os.path.exists(os.path.join(out, "test.csv")):
            parser.error("{} already has test.csv: the test match is scored once".format(out))
    run = os.path.basename(out)
    cache = os.path.join(config_lib.resolve_path(cfg["runtime"]["cache"]), run)
    logs = os.path.join(LOGS, run)
    os.makedirs(logs, exist_ok=True)
    say = logger(logs)
    torch.set_num_threads(cfg["runtime"]["threads"])
    say("run {} ({}), {} epochs, {} torch threads".format(
        run, "test" if args.test else "resume" if args.resume else args.config,
        cfg["train"]["epochs"], cfg["runtime"]["threads"]))

    data = Data(cfg)
    test = cfg["data"]["test_match"]
    dev = [name for name in data.matches if name != test]
    if args.test:
        run_test(data, test, dev, cfg, cache, out, say)
        return
    if args.bar_only:
        if cfg["bar"]["decoder"] == "grid":
            parser.error("[bar] decoder = grid needs the folds: run without --bar-only")
        run_bar(data, dev, cfg, cache, out, say)
        say("saved to {}, heads in {}".format(out, cache))
        return
    folds = cfg["data"]["folds"] or dev
    if test in folds:
        raise ValueError("the test match is never a development fold")

    epochs = []
    curve = os.path.join(logs, "train_log.csv")
    if os.path.exists(curve):
        epochs = pd.read_csv(curve).to_dict("records")

    def log_epoch(entry):
        epochs.append(entry)
        say("{fold}  epoch {epoch:3d}  train loss {train_loss:.4f}  train dead {train_dead_loss:.4f}"
            "  val dead {val_dead_loss:.4f}  dead_f1 {dead_f1:.3f}".format(**entry))
        pd.DataFrame(epochs).to_csv(curve, index=False, float_format="%.5f")

    run_folds(data, folds, dev, cfg, cache, say, log_epoch if cfg["train"]["log_every"] else None)
    table = score_folds(data, folds, cfg, cache)
    with_means(table).to_csv(os.path.join(out, "dev.csv"), index=False, float_format="%.4f")
    say("mean over {} folds, decoder {}:\n{}".format(
        len(folds), tuple(cfg["decoding"][k] for k in ("temperature", "w_start", "w_stop")),
        summary(table, cfg).to_string(float_format="{:.3f}".format)))

    if cfg["decoding"]["grid"]["enabled"]:
        from dead_alive_detection.decoding import grid
        _, text = grid.run_grid(cache, cfg, out, run, folds)
        say("decoder grid written to {}:\n{}".format(out, text))
    if cfg["bar"]["enabled"]:
        run_bar(data, dev, cfg, cache, out, say)
    say("saved to {}, logs in {}, heads in {}".format(out, logs, cache))


if __name__ == "__main__":
    main()
