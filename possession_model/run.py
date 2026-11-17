"""
Trains and scores the model leave-one-match-out: each DFL dev match is scored by a model
trained on the other 5. --test trains on all 6 and scores the test match once.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession_model import config, data, decode, evaluate, train  # noqa: E402

RESULTS = os.path.join(config.PACKAGE, "results")


def _new_run(name):
    i = 1
    while os.path.exists(os.path.join(RESULTS, "{}_{}".format(name, i))):
        i += 1
    return "{}_{}".format(name, i)


def oof_path(cache, fold, match, condition):
    return os.path.join(cache, "oof_{}_dfl_{}_{}.npz".format(fold, match, condition))


def model_path(cache, tag, seed):
    return os.path.join(cache, "model_{}_s{}.pt".format(tag, seed))


def write_scores(table, path):
    # Write to a temp file first so an interrupted run never leaves a half-written CSV
    temporary = path + ".tmp"
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


class Run:
    def __init__(self, cfg, run_name):
        self.cfg, self.name = cfg, run_name
        self.results = os.path.join(RESULTS, run_name)
        self.cache = os.path.join(config.resolve_path(cfg["runtime"]["cache"]), run_name)
        os.makedirs(self.results, exist_ok=True)
        os.makedirs(self.cache, exist_ok=True)
        torch.set_num_threads(cfg["runtime"]["threads"])

    def log(self, text):
        print(text, flush=True)
        with open(os.path.join(self.results, "train.log"), "a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def model_and_priors(self, train_names, tag, watch_match=None):
        # One model per seed plus the decoder priors. Models already on disk are loaded, not retrained.
        cfg = self.cfg
        items = train.training_items(cfg, [data.load(cfg, n) for n in train_names])
        priors = decode.Priors.from_items(items, cfg["steps"]["hz"])
        np.savez(os.path.join(self.cache, "priors_{}.npz".format(tag)), **priors.save())
        held = train.scored_item(cfg, watch_match, config.PRIMARY) if watch_match is not None else None
        models = []
        for seed in cfg["train"]["seeds"]:
            path = model_path(self.cache, tag, seed)
            model = train.PossessionModel(cfg).to(train.device(cfg))
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=train.device(cfg)))
                models.append(model)
                continue
            watch = None
            if held is not None:
                def watch(model, epoch):
                    logit, change = train.predict(cfg, model, held)
                    return {"heldout_spell_f1": round(self.score(priors, watch_match, held, logit, change)
                                                      ["spell_f1_mean"], 4)}
            self.log("training {} seed {} on {} matches, {} copies".format(tag, seed, len(train_names), len(items)))
            model = train.fit(cfg, items, seed, self.log, watch)
            torch.save(model.state_dict(), os.path.join(self.cache, "model_{}_s{}.pt".format(tag, seed)))
            pd.DataFrame(model.history).assign(fold=tag, seed=seed).to_csv(
                os.path.join(self.results, "train_log.csv"), mode="a", index=False,
                header=not os.path.exists(os.path.join(self.results, "train_log.csv")))
            models.append(model)
        return models, priors

    def outputs(self, models, it):
        # Average of the seeds
        outs = [train.predict(self.cfg, model, it) for model in models]
        return np.mean([o[0] for o in outs], axis=0), np.mean([o[1] for o in outs], axis=0)

    def score(self, priors, m, it, logit, change, dec=None):
        steps = decode.decode_steps(dec or self.cfg["decoding"], priors, logit, change, it["live"], it["period"])
        frames = decode.to_frames(steps, it["rows"], m.match.period, it["live_frames"])
        return evaluate.score(m.reference, frames)

    def score_held_out(self, models, priors, tag, m):
        # Scores every condition and saves the raw outputs so grid.py can re-decode without training
        rows = []
        for cond in config.CONDITIONS:
            it = train.scored_item(self.cfg, m, cond)
            logit, change = self.outputs(models, it)
            np.savez(oof_path(self.cache, tag, m.name, cond), logit=logit, change=change, live=it["live"],
                     period=it["period"], rows=it["rows"], live_frames=it["live_frames"])
            s = self.score(priors, m, it, logit, change)
            rows.append({"fold": tag, "dataset": "dfl", "match": m.name, "condition": cond, **s})
            self.log("  {} {}: spellF1 {:.3f}  team acc {:.3f}  changes F1@1s {:.3f}  possessions {}/{}".format(
                m.name, cond, s["spell_f1_mean"], s["team_acc"], s["trans_f1_1s"], s["n_pred"], s["n_gt"]))
        return rows

    def fold_complete(self, fold, scores):
        # Done only if all its scores, models, priors and outputs are on disk
        rows = scores.loc[scores["fold"] == fold]
        if (not {"dataset", "match", "condition"}.issubset(rows.columns)
                or len(rows) != len(config.CONDITIONS)
                or set(rows["condition"]) != set(config.CONDITIONS)
                or not (rows["dataset"] == "dfl").all()
                or not (rows["match"] == fold).all()):
            return False
        required = [model_path(self.cache, fold, seed) for seed in self.cfg["train"]["seeds"]]
        required.append(os.path.join(self.cache, "priors_{}.npz".format(fold)))
        required.extend(oof_path(self.cache, fold, fold, cond) for cond in config.CONDITIONS)
        return all(os.path.isfile(path) for path in required)

    def folds(self):
        dev, folds, _ = data.split(self.cfg)
        path = os.path.join(self.results, "dev.csv")
        scores = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["fold"])
        for fold in folds:
            if self.fold_complete(fold, scores):
                self.log("fold {}: done, skipped".format(fold))
                continue
            if (scores["fold"] == fold).any():
                self.log("fold {}: incomplete results or artifacts, rebuilding".format(fold))
                scores = scores.loc[scores["fold"] != fold].copy()
                # Drop the old scores now, so a crash during the rebuild doesn't look finished
                write_scores(scores, path)
            self.log("fold {}".format(fold))
            held = data.load(self.cfg, fold)
            models, priors = self.model_and_priors([n for n in dev if n != fold], fold, held)
            rows = self.score_held_out(models, priors, fold, held)
            fresh = pd.DataFrame(rows)
            scores = pd.concat([scores, fresh], ignore_index=True) if not scores.empty else fresh
            write_scores(scores, path)
        self.report()

    def test(self):
        dev, _, test = data.split(self.cfg)
        models, priors = self.model_and_priors(dev, "test")
        rows = self.score_held_out(models, priors, "test", data.load(self.cfg, test))
        pd.DataFrame(rows).to_csv(os.path.join(self.results, "test.csv"), index=False)
        lines = ["# {}: test match {}".format(self.name, test), "",
                 "Models trained on the {} development matches ({} seed(s)), scored once.".format(
                     len(dev), len(models)), "",
                 markdown_table(pd.DataFrame(rows)[["condition", *evaluate.KEYS]].round(4)), ""]
        with open(os.path.join(self.results, "test.md"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        print("\n".join(lines))

    def report(self):
        path = os.path.join(self.results, "dev.csv")
        if not os.path.exists(path):
            return
        dev = pd.read_csv(path)
        means = dev.groupby(["dataset", "condition"])[list(evaluate.KEYS)].mean().round(4)
        lines = ["# {}".format(self.name), "",
                 "Leave-one-match-out, mean over held-out matches. Headline condition: `{}`. "
                 "Reference: controlled DFL labels (min_frames = {}){}.".format(
                     config.PRIMARY, self.cfg["data"]["min_frames"],
                     " with the manual logs of `{}`".format(self.cfg["data"]["labels_dir"])
                     if self.cfg["data"]["labels_dir"] else ""), "",
                 markdown_table(means.reset_index()), ""]
        with open(os.path.join(self.results, "report.md"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        print("\n".join(lines))


def markdown_table(frame):
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def main(argv=None):
    parser = argparse.ArgumentParser(description="possession model: train and score, leave-one-match-out")
    parser.add_argument("--config", help="default: possession_model/configs/v1_frozen.toml")
    parser.add_argument("--set", action="append", default=[], help="section.key=value")
    parser.add_argument("--resume", help="an existing results/<run> folder")
    parser.add_argument("--test", action="store_true", help="score the test match (once, at the end)")
    args = parser.parse_args(argv)
    if args.resume:
        run_name = os.path.basename(os.path.normpath(args.resume))
        cfg = config.load(os.path.join(args.resume, "config.toml"), args.set)
    else:
        cfg = config.load(args.config, args.set)
        run_name = _new_run(cfg["name"])
    run = Run(cfg, run_name)
    config.dump(cfg, os.path.join(run.results, "config.toml"))
    if args.test:
        run.test()
    else:
        run.folds()


if __name__ == "__main__":
    main()
