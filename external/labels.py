"""Bring DFL possession labels closer to what our ground truth calls a possession.

DFL's BallPossession switches on touches our ground truth ignores: CHALLENGE.md says a touch
without control, a deflection, or an aerial duel won without follow-up is not a change of
possession. In DFL ~10-17% of possessions last under a second; in the public ground truth one of
seventeen does. Training on the raw flags would teach a model to produce exactly the flicker
spellF1 punishes.

controlled(label, period, min_frames) relabels every H/A spell shorter than min_frames that sits
between two spells of the other team -- H [A short] H -> H H H -- repeated until stable, since one
merge can create a new short sandwich. Spells next to a dead ball are left alone: a short spell
before a stoppage is often the team that put the ball out, which our ground truth keeps.

    python external/labels.py      # how possession counts and lengths move with min_frames

min_frames is a training hyper-parameter, not a fact. The table this prints is the evidence for
choosing it -- read it next to the public ground truth's 17 possessions per 5 min, mean 14.2 s,
and remember that seventeen possessions are a small sample.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from external import schema  # noqa: E402

# Params
CANDIDATES = (0, 12, 25, 50, 75, 100)
OTHER = {"H": "A", "A": "H"}


def controlled(label, period, min_frames):
    """Copy of label with short sandwiched possessions merged into the surrounding one."""
    label = np.array(label, copy=True)
    if min_frames <= 0:
        return label
    while True:
        spells = schema.spells(label, period)
        changed = False
        for i in range(1, len(spells) - 1):
            start, stop, team = spells[i]
            if team not in OTHER or stop - start >= min_frames:
                continue
            before, after = spells[i - 1], spells[i + 1]
            same_period = period[before[0]] == period[start] == period[after[0]]
            if before[2] == after[2] == OTHER[team] and same_period:
                label[start:stop] = OTHER[team]
                changed = True
        if not changed:
            return label


def main():
    paths = schema.processed_paths("dfl")
    if not paths:
        sys.exit("no processed DFL files -- run external/fetch_dfl.py first")
    matches = [schema.load_match(p) for p in paths]
    minutes = sum(len(m.frame) for m in matches) / schema.FPS / 60
    print("{} DFL matches; public ground truth: 17.0 poss/5 min, mean 14.2 s, 5.9% < 1 s\n".format(
        len(matches)))
    print("{:>10} {:>9} {:>8} {:>10} {:>7} {:>16}".format(
        "min_frames", "poss/5m", "mean s", "median s", "<1s", "frames changed"))
    for n in CANDIDATES:
        lengths, changed = [], 0
        for m in matches:
            new = controlled(m.label, m.period, n)
            changed += int((new != m.label).sum())
            lengths.append(schema.possession_lengths(new, m.period))
        lengths = np.concatenate(lengths)
        total = sum(len(m.frame) for m in matches)
        print("{:>10} {:>9.1f} {:>8.1f} {:>10.1f} {:>7.1%} {:>15.2%}".format(
            n, len(lengths) / minutes * 5, lengths.mean() / schema.FPS,
            np.median(lengths) / schema.FPS, np.mean(lengths < schema.FPS), changed / total))


if __name__ == "__main__":
    main()
