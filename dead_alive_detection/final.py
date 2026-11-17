"""
The final dead-ball model on BAR-ATH in one command: train every seed, average, decode, add the
tags.

    python -m dead_alive_detection.final

Trains tcn_bar_keeperhold for seeds 0-2 on the 7 DFL matches, decodes their mean probability with
the HSMM and runs the Smart Tagging post-step. Output goes to cache/dead_alive/<name>_tagchains/.
See README.md for the files it needs.
"""
import argparse
import os
import subprocess
import sys

from dead_alive_detection import config as config_lib
from dead_alive_detection.run import new_run_name
from external import schema

SEEDS = (0, 1, 2)
MODEL = "tcn_bar_keeperhold"
CONFIGS = "dead_alive_detection/configs"
NEEDED = {
    "data/processed/BAR-ATH.parquet": "python -m external.convert_atd",
    "cache/ball_loss/BAR-ATH.json": "python -m dead_alive_detection.augment.ball.fit_hidden",
    "cache/players_learned/BAR-ATH.json": "python -m dead_alive_detection.augment.players.fit_learned",
    "cache/dead_alive_dfl/bar_motion/library.npz": "python -m dead_alive_detection.augment.dead_ball.bar",
    "cache/audio/BAR-ATH_broadcast_whistles_preset.csv": "python -m dead_alive_detection.audio.whistle",
}


def seed_config(seed):
    """
    Path of the config that trains this seed's BAR model on the 7 DFL matches.
    """
    return "{}/{}{}_all7.toml".format(CONFIGS, MODEL, "_s{}".format(seed) if seed else "")


def missing_inputs(root):
    """
    Needed files that are missing, each with the command that builds it.
    """
    return {path: how for path, how in NEEDED.items() if not os.path.exists(os.path.join(root, path))}


def after_training(members, name, heads, teams=None, out=None):
    """
    Module calls that turn the seeds' heads folders into the BAR folders, and optionally a
    submission.
    """
    final_folder = "cache/dead_alive/{}_tagchains".format(os.path.basename(os.path.normpath(heads)))
    calls = [["dead_alive_detection.decoding.ensemble", "--out", heads, "--members"] + list(members),
             ["dead_alive_detection.predict", "--oof", heads, "--config", seed_config(SEEDS[0]),
              "--name", name],
             ["dead_alive_detection.analysis.tag_chains", "--bar-heads", heads,
              "--bar-config", seed_config(SEEDS[0])]]
    if out:
        calls.append(["dead_alive_detection.submit", "--dead", final_folder,
                      "--teams", teams, "--out", out])
    return calls


def call(args):
    print("$ python -m " + " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m"] + args, cwd=schema.ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--name", help="cache/dead_alive/<name> (default ens_keeperhold<number of seeds>_all7)")
    parser.add_argument("--teams", help="H/A csv for the submission")
    parser.add_argument("--out", help="submission CSV to write (keep it out of git)")
    args = parser.parse_args(argv)
    if bool(args.teams) != bool(args.out):
        parser.error("give --teams and --out together")
    if len(args.seeds) < 2:
        parser.error("an ensemble needs at least two seeds")
    missing = missing_inputs(schema.ROOT)
    if missing:
        parser.error("missing files, see README.md:\n" + "\n".join(
            "  {}  ({})".format(path, how) for path, how in missing.items()))

    name = args.name or "ens_keeperhold{}_all7".format(len(args.seeds))
    members = []
    for seed in args.seeds:
        config = seed_config(seed)
        cfg = config_lib.load(os.path.join(schema.ROOT, config))
        cache = config_lib.resolve_path(cfg["runtime"]["cache"])
        members.append(os.path.relpath(os.path.join(cache, new_run_name(cfg["name"])), schema.ROOT))
        call(["dead_alive_detection.run", "--config", config, "--bar-only"])
    heads = os.path.relpath(os.path.join(config_lib.resolve_path("cache/dead_alive_dfl"), name), schema.ROOT)
    for args_list in after_training(members, name, heads, args.teams, args.out):
        call(args_list)


if __name__ == "__main__":
    main()
