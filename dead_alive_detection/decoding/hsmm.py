"""
Two-state hidden semi-Markov decoder that turns per-step evidence into whole dead and alive segments.

Each segment is scored by its duration prior plus its summed evidence scaled by a temperature,
and Viterbi finds the best alternating sequence. This stops the flicker of a per-step threshold.
Optional start and stop evidence can reward segment boundaries.
"""

import numpy as np

ALIVE, DEAD = 0, 1


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


class HSMM:
    def __init__(self, log_pmf, temperature, w_start=0.0, w_stop=0.0):
        """
        log_pmf holds the duration log pmf of each state; w_start and w_stop weight the
        boundary evidence (0 means unused).
        """
        self.log_pmf = log_pmf
        self.temperature = temperature
        self.w_start, self.w_stop = w_start, w_stop
        with np.errstate(divide="ignore"):
            # log P(duration >= d): the reverse cumulative sum of the pmf.
            self.log_surv = {s: np.log(np.cumsum(np.exp(p)[::-1])[::-1]) for s, p in log_pmf.items()}
        # A censored FIRST segment is seen from a random point onwards: its seen length follows the
        # residual-life distribution, proportional to the survival function.
        self.log_first = {s: v - _logsumexp(v) for s, v in self.log_surv.items()}

    def _duration_scores(self, state, t, n):
        """
        Start and duration score of every segment of `state` ending at step t.
        """
        d = np.arange(1, min(len(self.log_pmf[state]), t) + 1)        # candidate lengths
        a = t - d
        if t == n:                             # the segment ends the half: censored on the right
            return a, np.where(a == 0, 0.0, self.log_surv[state][d - 1])
        return a, np.where(a == 0, self.log_first[state][d - 1], self.log_pmf[state][d - 1])

    def viterbi(self, llr, start_llr=None, stop_llr=None):
        """
        Dead mask of the most likely segmentation of one half.
        """
        n = len(llr)
        if n == 0:
            return np.zeros(0, dtype=bool)
        # cum[s][t]: summed evidence of state s over steps [0, t), so any segment costs O(1).
        cum = {ALIVE: np.zeros(n + 1),
               DEAD: np.r_[0.0, np.cumsum(self.temperature * np.nan_to_num(llr))]}
        # bonus_in[a]: earned by a dead segment starting at a; bonus_out[t]: by one ending at t.
        boundary = (self.w_start and start_llr is not None) or (self.w_stop and stop_llr is not None)
        if boundary:
            bonus_in, bonus_out = (
                np.r_[0.0, w * np.nan_to_num(x[1:]), 0.0] if w and x is not None else np.zeros(n + 1)
                for w, x in ((self.w_start, start_llr), (self.w_stop, stop_llr)))
        start = {s: np.full(n + 1, -np.inf) for s in (ALIVE, DEAD)}   # best score of paths that
        end = {s: np.full(n + 1, -np.inf) for s in (ALIVE, DEAD)}     # start / end a segment at t
        length = {s: np.zeros(n + 1, dtype=np.int64) for s in (ALIVE, DEAD)}
        start[ALIVE][0] = start[DEAD][0] = np.log(0.5)
        for t in range(1, n + 1):
            for s in (ALIVE, DEAD):
                a, prior = self._duration_scores(s, t, n)
                score = start[s][a] + prior + cum[s][t] - cum[s][a]
                if boundary and s == DEAD:
                    score = score + bonus_in[a] + bonus_out[t]
                best = int(np.argmax(score))
                end[s][t], length[s][t] = score[best], t - a[best]
            # States alternate: a dead segment starts where an alive one ends, and vice versa.
            start[DEAD][t], start[ALIVE][t] = end[ALIVE][t], end[DEAD][t]

        # Walk back from the end of the half, one segment at a time.
        dead = np.zeros(n, dtype=bool)
        state, t = (DEAD if end[DEAD][n] > end[ALIVE][n] else ALIVE), n
        while t > 0:
            d = int(length[state][t])
            dead[t - d:t] = state == DEAD
            state, t = 1 - state, t - d
        return dead

    def decode(self, llr, period, start_llr=None, stop_llr=None):
        """
        Dead mask for a whole match, decoding each half on its own.
        """
        dead = np.zeros(len(llr), dtype=bool)
        for p in (1, 2):
            rows = np.flatnonzero(period == p)
            dead[rows] = self.viterbi(llr[rows], *(None if x is None else x[rows]
                                                   for x in (start_llr, stop_llr)))
        return dead
