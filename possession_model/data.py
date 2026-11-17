"""
Which DFL matches we use (6 to develop, 1 for test) and their labels.
Labels = DFL's + our annotations, then short spells merged, except the ones we reviewed.
"""

import os

from external import labels as label_tools
from external import schema
from possession_model import config


def folder(cfg):
    return config.resolve_path(cfg["data"]["dfl_dir"])


def label_logs(cfg):
    # The annotation logs (records, bad lines). Nothing if labels_dir is empty.
    if not cfg["data"]["labels_dir"]:
        return [], []
    from possession_exploration import labelstore
    return labelstore.read_logs(config.resolve_path(cfg["data"]["labels_dir"]))


def labels_fingerprint(cfg):
    records, _ = label_logs(cfg)
    if not records:
        return ""
    from possession_exploration import labelstore
    return labelstore.labels_fingerprint(records, "dfl")


def names(cfg):
    if cfg["runtime"]["bundle"]:
        from possession_model import bundle
        return bundle.names(cfg)
    path = folder(cfg)
    if not os.path.isdir(path):
        return []
    return sorted(f[:-len(".parquet")] for f in os.listdir(path)
                  if f.startswith("DFL-MAT-") and f.endswith(".parquet"))


def split(cfg):
    # (dev matches, the folds to run, test match)
    every = names(cfg)
    if not every:
        raise FileNotFoundError("no DFL matches in {}: run external/fetch_dfl.py"
                                .format(folder(cfg)))
    test = cfg["data"]["test_match"]
    dev = [m for m in every if m != test]
    folds = cfg["data"]["folds"] or dev
    unknown = sorted(set(folds) - set(dev))
    if unknown:
        raise ValueError("data.folds: not development matches: {}".format(unknown))
    return dev, folds, test


class Match:
    # The match, its labels (target to train = reference to score) and its index, which seeds
    # the corruptions.

    def __init__(self, cfg, name):
        self.name = name
        self.match = schema.load_match(os.path.join(folder(cfg), name + ".parquet"))
        self.index = names(cfg).index(name)
        records, problems = label_logs(cfg)
        if problems:
            raise ValueError("malformed lines in the label logs: {}".format(problems))
        if records:
            from possession_exploration import labelstore
            labels = labelstore.model_labels(self.match.label, self.match.frame, self.match.period, records,
                                             "dfl", name, cfg["data"]["min_frames"])
        else:
            labels = label_tools.controlled(self.match.label, self.match.period, cfg["data"]["min_frames"])
        self.reference = self.target = labels


_CACHE = {}


def load(cfg, name):
    # Each match is read once per run. On Kaggle the labels come from the bundle.
    if cfg["runtime"]["bundle"]:
        from possession_model import bundle
        return bundle.load_match(cfg, name)
    key = (cfg["data"]["dfl_dir"], cfg["data"]["min_frames"], labels_fingerprint(cfg), name)
    if key not in _CACHE:
        _CACHE[key] = Match(cfg, name)
    return _CACHE[key]
