"""
Hidden semi-Markov decoder with one dead state per stoppage class.

Each class has its own duration prior and share, so a corner and a throw-in are no longer scored
alike. The TCN's type output can add evidence for the class of each stoppage. Otherwise it
works like hsmm.py.
"""

import numpy as np

from dead_alive_detection.decoding.hsmm import _logsumexp


def _censored(log_pmf):
    """
    Log survival and first-segment log pmf of a duration log pmf, as in hsmm.HSMM.
    """
    with np.errstate(divide="ignore"):
        surv = np.log(np.cumsum(np.exp(log_pmf)[::-1])[::-1])
    return surv, surv - _logsumexp(surv)


class TypedHSMM:
    def __init__(self, alive_log_pmf, dead_log_pmfs, shares, temperature, w_start=0.0, w_stop=0.0,
                 type_weight=0.0, evidence="mean"):
        """
        Takes the alive duration log pmf, one dead duration log pmf per class and the class shares.
        """
        self.log_pmf = [alive_log_pmf] + list(dead_log_pmfs)
        self.log_surv, self.log_first = zip(*[_censored(p) for p in self.log_pmf])
        shares = np.asarray(shares, dtype=float)
        with np.errstate(divide="ignore"):
            self.log_share = np.log(shares / shares.sum())
        self.temperature, self.w_start, self.w_stop = temperature, w_start, w_stop
        self.type_weight, self.evidence = type_weight, evidence
        self.n_types = len(dead_log_pmfs)

    def _duration_scores(self, s, t, n):
        d = np.arange(1, min(len(self.log_pmf[s]), t) + 1)
        a = t - d
        if t == n:
            return a, np.where(a == 0, 0.0, self.log_surv[s][d - 1])
        return a, np.where(a == 0, self.log_first[s][d - 1], self.log_pmf[s][d - 1])

    def viterbi(self, llr, start_llr=None, stop_llr=None, type_llr=None):
        """
        Dead mask and class per step (-1 when alive) for one half.
        """
        n, K = len(llr), self.n_types
        if n == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
        cum_dead = np.r_[0.0, np.cumsum(self.temperature * np.nan_to_num(llr))]
        bonus_in, bonus_out = (
            np.r_[0.0, w * np.nan_to_num(x[1:]), 0.0] if w and x is not None else np.zeros(n + 1)
            for w, x in ((self.w_start, start_llr), (self.w_stop, stop_llr)))
        use_type = self.type_weight and type_llr is not None
        if use_type:
            scale = self.temperature * self.type_weight if self.evidence == "sum" else 1.0
            cum_type = np.vstack([np.zeros(K), np.cumsum(scale * np.nan_to_num(type_llr), axis=0)])
        S = K + 1
        start = np.full((S, n + 1), -np.inf)
        end = np.full((S, n + 1), -np.inf)
        length = np.zeros((S, n + 1), dtype=np.int64)
        before = np.zeros(n + 1, dtype=np.int64)            # dead state that ended where ALIVE starts
        start[0, 0] = np.log(0.5)
        start[1:, 0] = np.log(0.5) + self.log_share
        for t in range(1, n + 1):
            for s in range(S):
                a, prior = self._duration_scores(s, t, n)
                score = start[s, a] + prior
                if s:
                    score = score + cum_dead[t] - cum_dead[a] + bonus_in[a] + bonus_out[t]
                    if use_type:
                        gain = cum_type[t, s - 1] - cum_type[a, s - 1]
                        score = score + (self.type_weight * gain / (t - a)
                                         if self.evidence == "mean" else gain)
                best = int(np.argmax(score))
                end[s, t], length[s, t] = score[best], t - a[best]
            k = int(np.argmax(end[1:, t]))
            before[t] = 1 + k
            start[0, t] = end[1 + k, t]
            start[1:, t] = end[0, t] + self.log_share

        dead = np.zeros(n, dtype=bool)
        kind = np.full(n, -1, dtype=np.int64)
        s, t = int(np.argmax(end[:, n])), n
        while t > 0:
            d = int(length[s, t])
            if s:
                dead[t - d:t], kind[t - d:t] = True, s - 1
            t -= d
            s = int(before[t]) if s == 0 else 0
        return dead, kind

    def decode(self, llr, period, start_llr=None, stop_llr=None, type_llr=None):
        """
        Dead mask and class per step for a whole match, decoding each half on its own.
        """
        dead = np.zeros(len(llr), dtype=bool)
        kind = np.full(len(llr), -1, dtype=np.int64)
        for p in (1, 2):
            rows = np.flatnonzero(period == p)
            parts = [None if x is None else x[rows] for x in (start_llr, stop_llr, type_llr)]
            dead[rows], kind[rows] = self.viterbi(llr[rows], *parts)
        return dead, kind
