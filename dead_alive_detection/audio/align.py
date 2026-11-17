"""
Convert audio times to BAR-ATH tactical video frames (the submission's frame index).

The tactical video's audio is in step with it. The broadcast is shorter and starts later, by a
constant offset per half.
"""

import numpy as np

FPS = 25
OFFSETS = (4014, 26920)     # tactical - broadcast frame, 1st and 2nd half
SPLIT = 86730               # broadcast frame where the 2nd-half offset takes over


def to_tactical(seconds, source):
    """
    Tactical frames of audio times (s) on the 'tactical' or 'broadcast' track.
    """
    frames = np.round(np.asarray(seconds, float) * FPS).astype(np.int64)
    if source == "tactical":
        return frames
    if source == "broadcast":
        return frames + np.where(frames < SPLIT, OFFSETS[0], OFFSETS[1])
    raise ValueError(f"unknown audio source {source!r}")


def to_broadcast(frames):
    """
    Broadcast times (s) of tactical frames, NaN where the broadcast has none.
    """
    frames = np.asarray(frames, np.int64)
    first = frames - OFFSETS[0]
    second = frames - OFFSETS[1]
    b = np.where(first < SPLIT, first, np.where(second >= SPLIT, second, -1)).astype(float)
    b[b < 0] = np.nan
    return b / FPS
