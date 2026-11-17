"""
Turns the network's per-step probabilities into whole possessions with an HSMM.
Each live stretch is decoded on its own; the possession lengths come from the training matches.
"""

import numpy as np

from dead_alive_detection.decoding.durations import log_duration_pmf, run_lengths
from dead_alive_detection.decoding.hsmm import HSMM


def _logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1 - p)))


class Priors:
    # What the decoder learns from the training matches: possession lengths and how often H and
    # changes appear.

    def __init__(self, lengths, p_h, p_change, hz):
        self.lengths, self.p_h, self.p_change, self.hz = np.asarray(lengths), p_h, p_change, hz

    @classmethod
    def from_items(cls, items, hz):
        lengths, ys, changes = [], [], []
        for item in items:
            y, live = item["y"], item["live"]
            scored = np.isfinite(y) & live
            runs = run_lengths(np.nan_to_num(y) > 0.5, item["period"], scored)
            lengths += runs[True] + runs[False]
            ys.append(y[scored])
            changes.append(item["change"][scored])
        ys, changes = np.concatenate(ys), np.concatenate(changes)
        return cls(lengths, float(np.mean(ys > 0.5)), float(np.mean(changes)), hz)

    def save(self):
        return {"lengths": self.lengths, "p_h": self.p_h, "p_change": self.p_change, "hz": self.hz}

    @classmethod
    def load(cls, d):
        return cls(d["lengths"], float(d["p_h"]), float(d["p_change"]), int(d["hz"]))

    def log_pmf(self, bandwidth, max_s):
        if not len(self.lengths):
            raise ValueError("no complete training possessions to learn durations from")
        return log_duration_pmf(self.lengths, int(max_s * self.hz), bandwidth)


def _episodes(live, period):
    change = np.flatnonzero((live[1:] != live[:-1]) | (period[1:] != period[:-1])) + 1
    return [(a, b) for a, b in zip(np.r_[0, change], np.r_[change, len(live)]) if live[a]]


def decode_steps(dec, prior, logit_h, change, live, period):
    """
    H / A on the live steps, D on the rest. Each possession scores log P(its length) +
    temperature * the network's evidence + w_change * the change head at its edges.
    """
    llr = -(np.asarray(logit_h, dtype=np.float64) - _logit(prior.p_h))     # evidence for A
    c_llr = np.asarray(change, np.float64) - _logit(prior.p_change)
    lp = prior.log_pmf(dec["bandwidth"], dec["max_s"])
    hsmm = HSMM({0: lp, 1: lp}, dec["temperature"], w_start=dec["w_change"], w_stop=dec["w_change"])
    labels = np.full(len(live), "D", dtype="<U1")
    for a, b in _episodes(live, period):
        is_a = hsmm.viterbi(llr[a:b], c_llr[a:b], c_llr[a:b])
        labels[a:b] = np.where(is_a, "A", "H")
    return labels


def to_frames(step_labels, rows, period_frames, live_frames):
    # Steps back to 25 fps: live frames take the team of the nearest step, the rest of each half
    # is D and everything outside the halves is X.
    n = len(period_frames)
    out = np.full(n, "X", dtype="<U1")
    halves = period_frames > 0
    out[halves] = "D"
    step_period = period_frames[rows]
    for p in (1, 2):
        steps = np.flatnonzero(step_period == p)
        frames = np.flatnonzero((period_frames == p) & live_frames)
        if not len(steps) or not len(frames):
            continue
        lab = step_labels[steps]
        known = np.isin(lab, ("H", "A"))
        if not known.any():
            continue
        # Dead steps borrow the team of the nearest live step, so a live frame next to a dead step
        # still gets a team
        idx = np.arange(len(lab))
        prev = np.maximum.accumulate(np.where(known, idx, -1))
        nxt = np.minimum.accumulate(np.where(known, idx, len(lab))[::-1])[::-1]
        use_prev = (prev >= 0) & ((nxt >= len(lab)) | (idx - prev <= nxt - idx))
        filled = np.where(use_prev, lab[np.maximum(prev, 0)], lab[np.minimum(nxt, len(lab) - 1)])
        at = rows[steps]
        k = np.clip(np.searchsorted(at, frames), 1, len(at) - 1) if len(at) > 1 else np.zeros(len(frames), int)
        if len(at) > 1:
            k = np.where(np.abs(frames - at[k - 1]) <= np.abs(at[k] - frames), k - 1, k)
        out[frames] = filled[k]
    return out
