"""
Explains why the HSMM kept each segment on BAR-ATH.

    python -m dead_alive_detection.decoding.explain --run <results/run>

For each inner segment, the score for keeping it rather than merging it into its neighbours is
split into duration prior, TCN evidence and boundaries. A small margin is a close call.
Writes decisions.csv into the BAR folder. Two-state decoder only.
"""

import argparse
import os

import numpy as np
import pandas as pd

from dead_alive_detection.decoding.hsmm import ALIVE, DEAD

COLUMNS = ["start", "stop", "seconds", "half", "state", "mean_p_dead", "prior", "tcn", "boundary",
           "margin"]


def _duration(hsmm, state, a, b, n):
    """
    Duration score of segment [a, b) in a half of n steps, as Viterbi computes it.
    """
    d = b - a
    if d > len(hsmm.log_pmf[state]):
        return -np.inf
    if b == n:
        return 0.0 if a == 0 else hsmm.log_surv[state][d - 1]
    return hsmm.log_first[state][d - 1] if a == 0 else hsmm.log_pmf[state][d - 1]


def _dead_terms(hsmm, llr, start_llr, stop_llr, a, b, n):
    """
    Evidence and boundary scores of dead segment [a, b).
    """
    evidence = hsmm.temperature * float(np.nan_to_num(llr[a:b]).sum())
    boundary = 0.0
    if hsmm.w_start and start_llr is not None and a > 0:
        boundary += hsmm.w_start * float(np.nan_to_num(start_llr[a]))
    if hsmm.w_stop and stop_llr is not None and b < n:
        boundary += hsmm.w_stop * float(np.nan_to_num(stop_llr[b]))
    return evidence, boundary


def half_decisions(hsmm, dead, llr, start_llr=None, stop_llr=None):
    """
    Score breakdown of each inner segment of one half, in steps.
    """
    n = len(dead)
    change = np.flatnonzero(dead[1:] != dead[:-1]) + 1
    segs = list(zip(np.r_[0, change], np.r_[change, n]))
    rows = []
    for (a1, _), (a2, b2), (_, b3) in zip(segs, segs[1:], segs[2:]):
        middle, outer = (DEAD, ALIVE) if dead[a2] else (ALIVE, DEAD)
        prior = (_duration(hsmm, outer, a1, a2, n)
                 + _duration(hsmm, middle, a2, b2, n) + _duration(hsmm, outer, b2, b3, n)
                 - _duration(hsmm, outer, a1, b3, n))
        if middle == DEAD:          # keep the stoppage: its own evidence and boundaries
            tcn, boundary = _dead_terms(hsmm, llr, start_llr, stop_llr, a2, b2, n)
        else:                       # keep the gap: two stoppages against one over it
            e1, b1 = _dead_terms(hsmm, llr, start_llr, stop_llr, a1, a2, n)
            e3, bb3 = _dead_terms(hsmm, llr, start_llr, stop_llr, b2, b3, n)
            e, bb = _dead_terms(hsmm, llr, start_llr, stop_llr, a1, b3, n)
            tcn, boundary = e1 + e3 - e, b1 + bb3 - bb
        rows.append((a2, b2, middle, prior, tcn, boundary))
    return rows


def decisions(decoder, logits, prior, period, row):
    """
    Table of decisions of a two-state decoder on a match, in frames.
    """
    llr, start_llr, stop_llr = decoder.evidence(logits, prior)
    dead = decoder.decode(logits, prior, period)
    p = 1 / (1 + np.exp(-logits[:, 0]))
    step = int(np.median(np.diff(row))) if len(row) > 1 else 1
    out = []
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        for a, b, state, pr, tcn, bd in half_decisions(decoder.hsmm, dead[rows], llr[rows],
                                                       start_llr[rows], stop_llr[rows]):
            first, last = int(row[rows[a]]), int(row[rows[b - 1]]) + step
            out.append({"start": first, "stop": last, "seconds": (b - a) * step / 25, "half": half,
                        "state": "D" if state == DEAD else "alive",
                        "mean_p_dead": round(float(np.nanmean(p[rows[a:b]])), 3),
                        "prior": round(pr, 2), "tcn": round(tcn, 2), "boundary": round(bd, 2),
                        "margin": round(pr + tcn + bd, 2)})
    return pd.DataFrame(out, columns=COLUMNS)


def write(out, decoder, bar):
    """
    Writes decisions.csv into a BAR folder; does nothing for a typed decoder.
    """
    from dead_alive_detection.decoding.decoder import TypedDecoder
    if isinstance(decoder, TypedDecoder):     # its segments come from the typed HSMM's states
        return None
    table = decisions(decoder, bar["logits"], bar["prior"], bar["period"], bar["row"])
    table.to_csv(os.path.join(out, "decisions.csv"), index=False)
    return table


def main(argv=None):
    from dead_alive_detection import config as config_lib
    from dead_alive_detection.decoding import grid
    from dead_alive_detection.predict import DEAD_ALIVE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="results/<run> folder")
    parser.add_argument("--oof", help="folder with bar.npz (default: the run's)")
    parser.add_argument("--config", help="config (default: the run's)")
    parser.add_argument("--name", help="cache/dead_alive/<name> (default: [bar] name, else the run)")
    parser.add_argument("--decoder", choices=("config", "grid"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--frames", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="print the segments overlapping these frames")
    args = parser.parse_args(argv)
    if not args.run and not (args.oof and args.config):
        parser.error("give --run, or --oof and --config")
    cfg = config_lib.load(args.config or os.path.join(args.run, "config.toml"), args.set)
    run_name = os.path.basename(os.path.normpath(args.run or args.oof))
    folder = args.oof or os.path.join(config_lib.resolve_path(cfg["runtime"]["cache"]), run_name)
    bar = dict(np.load(os.path.join(folder, "bar.npz")))
    setting = None
    if (args.decoder or cfg["bar"]["decoder"]) == "grid":
        setting = grid.choice(grid.read_grid(os.path.join(args.run, "grid.csv")), cfg)
    decoder = grid.decoder_for(list(bar["train"]), cfg, setting)
    # As predict.write_folder: only the steps inside the halves (from each kickoff).
    from dead_alive_detection.loading.clean import played
    from external import schema
    match = played(schema.load_match(config_lib.resolve_path(cfg["bar"]["match"])))
    bar["period"] = np.where(match.period[bar["row"]] > 0, bar["period"], 0)
    out = os.path.join(DEAD_ALIVE, args.name or cfg["bar"]["name"] or run_name)
    table = write(out, decoder, bar)
    if table is None:
        print("typed decoder: not explained (two-state decoder only)")
        return
    close = table[table["margin"] < 10]
    print("{}: {} decisions, {} close (margin < 10); median for keeping: prior {:+.1f}, tcn {:+.1f}, "
          "boundary {:+.1f}".format(os.path.join(out, "decisions.csv"), len(table), len(close),
                                    table["prior"].median(), table["tcn"].median(),
                                    table["boundary"].median()))
    if args.frames:
        a, b = args.frames
        print(table[(table["stop"] > a) & (table["start"] < b)].to_string(index=False))


if __name__ == "__main__":
    main()
