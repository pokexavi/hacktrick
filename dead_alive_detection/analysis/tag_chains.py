"""
Turn empty Smart Tagging set pieces into dead balls that last as long as P(dead) stays up (BAR-ATH
only).

    python -m dead_alive_detection.analysis.tag_chains

In a set-piece tag with no decoded stoppage near it, the bumps of P(dead) above THR that touch the
tag become one dead ball, bridging dips shorter than GAP_S. GAP_S is the median true DFL stoppage of
that tag type. Writes cache/dead_alive/<bar heads>_tagchains/.
"""

import argparse
import os

import numpy as np

from dead_alive_detection import config as config_lib
from dead_alive_detection.analysis.tag_alignment import set_pieces
from dead_alive_detection.decoding import grid as gridlib
from dead_alive_detection.decoding.decoder import segment_classes
from dead_alive_detection.evaluation import metrics
from dead_alive_detection.evaluation.metrics import pm
from dead_alive_detection.evaluation.stoppage_types import fine_stoppages
from dead_alive_detection.loading.clean import played
from dead_alive_detection.loading.tensors import STEPS_PER_S
from dead_alive_detection.predict import DEAD_ALIVE, write_folder
from external import schema

THR = 0.05
MARGIN_S = 2.0                      # a dead step this close to a tag counts as inside it
BAR_HEADS = "cache/dead_alive_dfl/ens_keeperhold3_all7"
BAR_CONFIG = "dead_alive_detection/configs/tcn_bar_keeperhold_all7.toml"
FPS = schema.FPS
S = STEPS_PER_S
# Smart Tagging set-piece type -> the DFL stoppage class (evaluation/stoppage_types.fine_class) whose
# median duration is its GAP_S. A free kick restarts a stoppage the referee whistled.
TAG_CLASS = {"throw_in": "throw_in", "goal_kick": "goal_kick", "corner": "corner",
             "free_kick": "whistle"}


class Variant:
    """
    Two-state decoder whose output is changed by a post-step.
    """

    def __init__(self, base, post):
        self.base, self.hsmm, self.setting, self._post = base, base.hsmm, base.setting, post

    def evidence(self, logits, prior):
        return self.base.evidence(logits, prior)

    def decode(self, logits, prior, period):
        dead, start, stop = self.evidence(logits, prior)
        return self._post(self.hsmm.decode(dead, period, start, stop), logits, period)

    def decode_types(self, logits, prior, period):
        dead = self.decode(logits, prior, period)
        return dead, segment_classes(dead, period, logits)


def runs(mask):
    """
    (a, b) runs of True.
    """
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def step_windows(windows, rows):
    """
    Step ranges (lo, hi) of frame windows; rows are the steps' frames.
    """
    return [(int(np.searchsorted(rows, a, side="left")), int(np.searchsorted(rows, b, side="left")))
            for a, b in windows]


def dev_matches(cfg):
    """
    Names of the DFL development matches: every DFL match except the test match.
    """
    names = [os.path.basename(p).split(".")[0] for p in schema.processed_paths("dfl")]
    return [n for n in names if n.startswith("DFL-") and n != cfg["data"]["test_match"]]


def stoppage_medians(cfg):
    """
    Median duration (s) of the true DFL stoppages per tag type, over the development matches.
    """
    durations = {c: [] for c in TAG_CLASS.values()}
    for name in dev_matches(cfg):
        match = gridlib.dfl_match(name)
        for a, b, kind in fine_stoppages(match, pm.label_codes(metrics.truth_labels(match))):
            if kind in durations:
                durations[kind].append((b - a) / FPS)
    return {tag: float(np.median(durations[c])) for tag, c in TAG_CLASS.items()}


def fill(dead, p, period, windows, thr, gap, margin):
    """
    Add dead balls inside the windows (in steps) to the dead mask. Returns the new mask and the
    added (a, b) dead balls.
    """
    out = dead.copy()
    n = len(dead)
    added = []
    gaps = list(gap) if np.ndim(gap) else [gap] * len(windows)
    for (lo, hi), gap in zip(windows, gaps):
        if hi <= lo or dead[max(0, lo - margin):min(n, hi + margin)].any():
            continue
        for half in np.unique(period[lo:hi]):
            idx = np.flatnonzero(period == half)
            h0, h1 = int(idx[0]), int(idx[-1]) + 1           # the half's steps
            ups = [(h0 + a, h0 + b) for a, b in runs((p[h0:h1] > thr) | dead[h0:h1])]
            chains, k = [], 0
            while k < len(ups):                      # chains of bumps across dips shorter than gap
                first = k
                while k + 1 < len(ups) and ups[k + 1][0] - ups[k][1] < gap:
                    k += 1
                chains.append(ups[first:k + 1])
                k += 1
            for chain in chains:
                if not any(a < hi and b > lo for a, b in chain):     # a bump of it overlaps the tag
                    continue
                a, b = chain[0][0], chain[-1][1]
                if not out[a:b].all():
                    added.append((a, b))
                out[a:b] = True
    return out, added


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-heads", default=BAR_HEADS)
    parser.add_argument("--bar-config", default=BAR_CONFIG)
    parser.add_argument("--thr", type=float, default=THR)
    args = parser.parse_args(argv)
    cfg = config_lib.load(config_lib.resolve_path(args.bar_config))
    gap_s = stoppage_medians(cfg)
    folder = config_lib.resolve_path(args.bar_heads)
    b = dict(np.load(os.path.join(folder, "bar.npz")))
    match = played(schema.load_match(config_lib.resolve_path(cfg["bar"]["match"])))
    write(folder, b, match, cfg, gridlib.decoder_for(list(b["train"]), cfg), gap_s, args.thr)


def write(folder, b, match, cfg, base, gap_s, thr=THR):
    """
    Write cache/dead_alive/<heads folder>_tagchains/ from the BAR heads, the run's decoder and
    gap_s.
    """
    frames = match.frame[b["row"]]
    tags = [t for t in set_pieces() if t["kind"] != "kickoff"]
    windows = step_windows([(t["start"], t["stop"]) for t in tags], frames)
    gaps = [round(gap_s[t["kind"]] * S) for t in tags]
    p = 1 / (1 + np.exp(-b["logits"][:, 0].astype(float)))
    report = {}

    def post(dead, logits, period):
        out, added = fill(dead, p, period, windows, thr, gaps, round(MARGIN_S * S))
        report["added"] = [(int(frames[a]), int(frames[b_ - 1]), round((b_ - a) / S, 1)) for a, b_ in added]
        m = round(MARGIN_S * S)
        report["empty"] = [(t["kind"], int(t["start"])) for (lo, hi), t in zip(windows, tags)
                           if hi > lo and not dead[max(0, lo - m):hi + m].any()]
        return out
    decoder = Variant(base, post=post)
    name = os.path.basename(os.path.normpath(folder))
    out = write_folder(os.path.join(DEAD_ALIVE, name + "_tagchains"), match, b, decoder, cfg,
                       {"heads": os.path.relpath(folder, schema.ROOT).replace("\\", "/"),
                        "variant": "tagchains", "thr": thr, "gap_s": gap_s})
    print("empty tags (kind, first frame):", report.get("empty"))
    print("thr {}, gaps {}; dead balls added (first frame, last frame, seconds): {}".format(
        thr, {k: round(v, 1) for k, v in gap_s.items()}, report.get("added")))
    return out


if __name__ == "__main__":
    main()
