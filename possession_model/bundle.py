"""
Packs the code and the already-corrupted DFL copies into a zip to train on Kaggle.
Nothing from BAR goes in: only DFL positions, DFL labels (with the annotations) and the code.
"""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from types import SimpleNamespace

import numpy as np

from possession_model import config

CODE = ["possession_model/__init__.py", "possession_model/config.py", "possession_model/data.py",
        "possession_model/augment.py", "possession_model/features.py", "possession_model/model.py",
        "possession_model/train.py", "possession_model/decode.py", "possession_model/evaluate.py",
        "possession_model/run.py", "possession_model/grid.py", "possession_model/bundle.py",
        "possession_model/configs/v1_frozen.toml",
        "possession_model/README.md",
        "external/__init__.py", "external/schema.py", "external/labels.py",
        "evaluation/possession_metrics.py",
        "dead_alive_detection/decoding/hsmm.py", "dead_alive_detection/decoding/durations.py"]
EMPTY_PACKAGES = ["dead_alive_detection/__init__.py", "dead_alive_detection/decoding/__init__.py"]
REQUIREMENTS = """# Kaggle images already have torch (with CUDA), numpy and pandas. Only on Python < 3.11:
tomli
"""


def item_settings(cfg):
    # If any of these change, the copies in the bundle are no longer valid
    aug = cfg["augment"]
    return {"dfl_dir": cfg["data"]["dfl_dir"], "labels_dir": cfg["data"]["labels_dir"],
            "min_frames": cfg["data"]["min_frames"], "steps": cfg["steps"],
            "hidden_fracs": aug["hidden_fracs"],
            "fragments": aug["fragments"], "learned": aug["learned"], "ball": aug["ball"],
            "change_sigma_s": cfg["heads"]["change_sigma_s"]}


def item_name(recipe, copy, frac):
    return "{}__c{}__{}.npz".format(recipe, copy, "bar" if frac is None else repr(float(frac)))


# Reading (on Kaggle)

_MANIFESTS = {}


def manifest(cfg):
    root = cfg["runtime"]["bundle"]
    if root not in _MANIFESTS:
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as handle:
            man = json.load(handle)
        if json.loads(json.dumps(man.get("item_settings"))) != json.loads(json.dumps(item_settings(cfg))):
            raise ValueError("this bundle was made with other label / step / corruption settings: "
                             "change them back, or export a new bundle")
        _MANIFESTS[root] = man
    return _MANIFESTS[root]


def names(cfg):
    return manifest(cfg)["matches"]["dfl"]


def load_match(cfg, name):
    # The same fields data.Match has, read from the bundle
    with np.load(os.path.join(cfg["runtime"]["bundle"], "labels", "dfl", name + ".npz")) as f:
        return SimpleNamespace(name=name, index=int(f["index"]), reference=f["reference"], target=f["target"],
                               match=SimpleNamespace(period=f["period"], frame=f["frame"]))


def load_item(cfg, name, recipe, copy, frac):
    path = os.path.join(cfg["runtime"]["bundle"], "dfl", name, item_name(recipe, copy, frac))
    if not os.path.exists(path):
        raise FileNotFoundError("{} is not in the bundle: export a new one".format(path))
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# Writing (on your machine)

def export(cfg, out_zip, code_only=False):
    stage = tempfile.mkdtemp(prefix="possession_bundle_")
    top = os.path.join(stage, "possession_kaggle")
    items_dir, code_dir = os.path.join(top, "items"), os.path.join(top, "code")
    if not code_only:
        _export_items(cfg, items_dir)
    _export_code(code_dir)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(top):
            for f in files:
                path = os.path.join(folder, f)
                kind = zipfile.ZIP_STORED if f.endswith(".npz") else zipfile.ZIP_DEFLATED
                z.write(path, os.path.relpath(path, stage), compress_type=kind)
    shutil.rmtree(stage)
    print("{}: {}, {:.1f} MB".format(out_zip, "code only" if code_only else "code and copies",
                                     os.path.getsize(out_zip) / 1e6))


def _export_items(cfg, items_dir):
    from possession_model import augment, data, features, train
    if cfg["runtime"]["bundle"]:
        raise ValueError("export computes the copies: runtime.bundle must be empty")
    dev, _, test = data.split(cfg)
    corruptor = augment.Corruptor(cfg)
    every = sorted(dev + [test])
    jobs = [(n, spec) for n in dev for spec in train.training_specs(cfg)]
    jobs += [(n, (c, 0, None)) for n in every for c in config.CONDITIONS]
    for k, (name, (recipe, copy, frac)) in enumerate(jobs, 1):
        m = data.load(cfg, name)
        version, live = corruptor.corrupt(m, recipe, copy, frac)
        it = features.build(cfg, version, live, m.target)
        folder = os.path.join(items_dir, "dfl", name)
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(os.path.join(folder, item_name(recipe, copy, frac)), **it)
        print("  [{}/{}] {} {} copy {}".format(k, len(jobs), name, recipe, copy), flush=True)
    os.makedirs(os.path.join(items_dir, "labels", "dfl"), exist_ok=True)
    for name in every:
        m = data.load(cfg, name)
        np.savez_compressed(os.path.join(items_dir, "labels", "dfl", name + ".npz"),
                            reference=m.reference, target=m.target, period=m.match.period,
                            frame=m.match.frame, index=m.index)
    with open(os.path.join(items_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"config": "v1_frozen.toml", "item_settings": item_settings(cfg),
                   "labels_fingerprint": data.labels_fingerprint(cfg),     # which annotations it has
                   "matches": {"dfl": every}, "dev": dev, "test": test, "bar_pieces": False}, handle, indent=2)


def _export_code(code_dir):
    for rel in CODE:
        dest = os.path.join(code_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(config.resolve_path(rel), dest)
    for rel in EMPTY_PACKAGES:
        open(os.path.join(code_dir, rel), "w").close()
    with open(os.path.join(code_dir, "requirements_kaggle.txt"), "w") as handle:
        handle.write(REQUIREMENTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description="export a training bundle (no file of the provided match)")
    parser.add_argument("--out", required=True, help="the zip to write")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--code-only", action="store_true",
                        help="only code/: to update the code of a bundle whose items/ are already uploaded")
    args = parser.parse_args(argv)
    export(config.load(None, args.set), args.out, args.code_only)


if __name__ == "__main__":
    main()
