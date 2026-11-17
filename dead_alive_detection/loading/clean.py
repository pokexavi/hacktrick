"""
Cleans label artefacts before a match is used for training or scoring.

Some DFL halves start or end with a very short D run while the recording switches on or off.
These are not real stoppages, so they are relabelled X: neither trained on nor scored.
"""

from dataclasses import replace

import numpy as np

from external import schema

# Params
MAX_EDGE_ARTEFACT_FRAMES = 50   # 2 s: the longest artefact seen in DFL is 1.9 s; stoppages are longer
SECOND_HALF_KICKOFF = 113742    # BAR-ATH video frame (FAQ.md)


def played(match):
    """
    The match with BAR-ATH's second-half frames before kickoff moved outside the halves (period 0).
    A DFL match is returned unchanged.
    """
    if match.source == "dfl":
        return match
    period = match.period.copy()
    period[(period == 2) & (match.frame < SECOND_HALF_KICKOFF)] = 0
    return replace(match, period=period)


def mask_edge_artifacts(label, period, max_frames=MAX_EDGE_ARTEFACT_FRAMES):
    """
    Copy of label with the short D runs at the start or end of a half set to X.
    """
    label = label.copy()
    # Half edges as run boundaries: a run starts a half if it starts on the half's first frame,
    # and ends it if it stops one past the half's last frame (spells are half-open).
    halves = [np.flatnonzero(period == p) for p in (1, 2)]
    edges = {int(rows[0]) for rows in halves if len(rows)} | \
            {int(rows[-1]) + 1 for rows in halves if len(rows)}
    for start, stop, value in schema.spells(label, period):
        if value == "D" and stop - start < max_frames and (start in edges or stop in edges):
            label[start:stop] = "X"
    return label
