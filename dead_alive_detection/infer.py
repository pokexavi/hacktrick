"""
The final dead-ball model on BAR-ATH from the saved weights: no DFL data, no training.

    python -m dead_alive_detection.infer

Runs the same steps as final.py after training, using the weights and decoder.npz in
dead_alive_detection/weights/. Needs the competition data (see DATA.md). Writes
cache/dead_alive/ens_keeperhold3_all7_tagchains/.
"""

import argparse
import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import torch

from dead_alive_detection import config as config_lib
from dead_alive_detection.final import SEEDS, seed_config
from external import schema

PACKAGE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(PACKAGE, "weights")
NAME = "ens_keeperhold3_all7"
# The runs whose BAR models were submitted (cache/dead_alive_dfl/ens_keeperhold3_all7/members.json).
RUNS = ["cache/dead_alive_dfl/tcn_bar_keeperhold_all7_1",
        "cache/dead_alive_dfl/tcn_bar_keeperhold_s1_all7_1",
        "cache/dead_alive_dfl/tcn_bar_keeperhold_s2_all7_1"]


def member(seed):
    """
    Model name of a seed: its config's file name.
    """
    return os.path.basename(seed_config(seed))[:-len(".toml")]


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_sums(folder):
    names = sorted(f for f in os.listdir(folder) if f.endswith((".pt", ".npz")))
    with open(os.path.join(folder, "SHA256SUMS"), "w", newline="\n") as handle:
        handle.writelines("{}  {}\n".format(sha256(os.path.join(folder, n)), n) for n in names)


def check_sums(folder):
    with open(os.path.join(folder, "SHA256SUMS")) as handle:
        for line in handle:
            digest, name = line.split()
            if sha256(os.path.join(folder, name)) != digest:
                raise ValueError("{}: checksum differs from SHA256SUMS".format(name))


def save_decoder(path, names, steps, gap_s):
    """
    Save each training match's (dead, period, scored) steps and gap_s to decoder.npz.
    """
    kinds = sorted(gap_s)
    arrays = {"names": np.array(names), "gap_kinds": np.array(kinds),
              "gap_s": np.array([gap_s[k] for k in kinds], dtype=np.float64)}
    for i, (dead, period, scored) in enumerate(steps):
        arrays.update({"dead_{}".format(i): dead, "period_{}".format(i): period,
                       "scored_{}".format(i): scored})
    np.savez_compressed(path, **arrays)


def load_decoder(path):
    """
    Read a decoder.npz back as (names, steps, gap_s).
    """
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    steps = [(z["dead_{}".format(i)], z["period_{}".format(i)], z["scored_{}".format(i)])
             for i in range(len(names))]
    if "gap_kinds" not in z:
        raise ValueError("{}: no gap per tag type, re-export it (infer --export)".format(path))
    return names, steps, {str(k): float(v) for k, v in zip(z["gap_kinds"], z["gap_s"])}


def export(runs, folder):
    """
    Refresh weights/ from the trained runs. Needs the DFL matches.
    """
    from dead_alive_detection.analysis.tag_chains import stoppage_medians
    from dead_alive_detection.decoding import grid
    os.makedirs(folder, exist_ok=True)
    trains = []
    for seed, run in zip(SEEDS, runs):
        run = config_lib.resolve_path(run)
        shutil.copyfile(os.path.join(run, "models", "BAR-ATH.pt"),
                        os.path.join(folder, member(seed) + ".pt"))
        trains.append([str(n) for n in np.load(os.path.join(run, "bar.npz"))["train"]])
    if any(t != trains[0] for t in trains):
        raise ValueError("the runs' BAR heads were decoded with different matches: {}".format(trains))
    cfg = config_lib.load(config_lib.resolve_path(seed_config(SEEDS[0])))
    save_decoder(os.path.join(folder, "decoder.npz"), trains[0], grid.labelled_steps(trains[0]),
                 stoppage_medians(cfg))
    write_sums(folder)
    print("exported {} models and decoder.npz ({} DFL matches) to {}".format(
        len(runs), len(trains[0]), folder))


def bar_heads(trained, prior, cfg, match, names, path):
    """
    A seed's bar.npz, as run.run_bar saves it.
    """
    from dead_alive_detection.augment.views import inputs
    from dead_alive_detection.loading.tensors import match_tensors
    from dead_alive_detection.run import save_heads
    t = match_tensors(match)                       # BAR's ball as it is: it is already lost
    if cfg["features"]["whistle"]:                 # the detector's whistles on the broadcast audio
        from dead_alive_detection.augment.whistle import simulated
        table = pd.read_csv(config_lib.resolve_path(cfg["bar"]["whistles"]))
        t["whistle"] = simulated.to_steps(simulated.detected(table, match.frame), t["row"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_heads(path, t, trained.heads(inputs(t, cfg), t), prior, names)


def infer(name, folder):
    from dead_alive_detection.analysis import tag_chains
    from dead_alive_detection.decoding import ensemble
    from dead_alive_detection.decoding.decoder import make_decoder
    from dead_alive_detection.loading.clean import played
    from dead_alive_detection.predict import DEAD_ALIVE, write_folder
    from dead_alive_detection.training.fit_tcn import Trained
    check_sums(folder)
    names, steps, gap_s = load_decoder(os.path.join(folder, "decoder.npz"))
    cfg = config_lib.load(config_lib.resolve_path(seed_config(SEEDS[0])))
    if cfg["decoding"]["typed"]["enabled"]:
        raise ValueError("decoder.npz holds no stoppage types: the final model's decoder is two-state")
    missing = [p for p in (cfg["bar"]["match"], cfg["bar"]["whistles"])
               if not os.path.exists(config_lib.resolve_path(p))]
    if missing:
        raise SystemExit("missing, see README.md (Reproduction): " + ", ".join(missing))
    torch.set_num_threads(cfg["runtime"]["threads"])
    match = played(schema.load_match(config_lib.resolve_path(cfg["bar"]["match"])))
    heads = os.path.join(config_lib.resolve_path(cfg["runtime"]["cache"]), name)
    members = []
    for seed in SEEDS:
        seed_cfg = config_lib.load(config_lib.resolve_path(seed_config(seed)))
        trained, extra = Trained.load(os.path.join(folder, member(seed) + ".pt"))
        if extra["features"] != seed_cfg["features"]:
            raise ValueError("{}: saved [features] differ from {}".format(member(seed), seed_config(seed)))
        members.append(os.path.join(heads, "members", member(seed)))
        bar_heads(trained, extra["prior"], seed_cfg, match, names, os.path.join(members[-1], "bar.npz"))
        print("{}: heads on BAR-ATH".format(member(seed)), flush=True)
    ensemble.write(heads, members)
    b = dict(np.load(os.path.join(heads, "bar.npz")))
    relative = os.path.relpath(heads, schema.ROOT).replace("\\", "/")
    write_folder(os.path.join(DEAD_ALIVE, name), match, b, make_decoder(steps, None, cfg), cfg,
                 {"heads": relative})
    tag_chains.write(heads, b, match, cfg, make_decoder(steps, None, cfg), gap_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=NAME,
                        help="output folders cache/dead_alive/<name>[_tagchains] (default %(default)s)")
    parser.add_argument("--weights", default=WEIGHTS, help="folder of the weights (default %(default)s)")
    parser.add_argument("--export", action="store_true",
                        help="write --weights from the trained runs (needs the DFL data)")
    parser.add_argument("--runs", nargs=len(SEEDS), default=RUNS, help="with --export: the seeds' runs")
    args = parser.parse_args(argv)
    if args.export:
        export(args.runs, args.weights)
    else:
        infer(args.name, args.weights)


if __name__ == "__main__":
    main()
