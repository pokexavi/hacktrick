"""
Turns the TCN's heads into a 5 Hz dead mask for one fold.

Duration and boundary priors are learnt from the fold's training matches. Each head's logit
minus its training prior is fed to the HSMM as evidence. With [decoding.typed] enabled,
make_decoder returns a TypedDecoder with one dead state per stoppage class.
"""

import numpy as np

from dead_alive_detection.decoding.durations import log_duration_pmf, run_lengths  # noqa: F401
from dead_alive_detection.decoding.hsmm import ALIVE, DEAD, HSMM
from dead_alive_detection.decoding.typed import TypedHSMM
from dead_alive_detection.loading.tensors import STEPS_PER_S
from dead_alive_detection.training.targets import targets


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def labelled(tensors):
    """
    The (dead, period, scored) arrays of each match, all the decoder needs from training.
    """
    return [(t["dead"], t["period"], t["scored"]) for t in tensors]


def durations(labelled_steps, cfg):
    """
    Duration log pmf of alive and dead runs in the given matches.
    """
    runs = {True: [], False: []}
    for dead, period, scored in labelled_steps:
        for state, lengths in run_lengths(dead, period, scored).items():
            runs[state] += lengths
    return {DEAD: log_duration_pmf(runs[True], round(cfg["dead_max_s"] * STEPS_PER_S),
                                   cfg["bandwidth"]),
            ALIVE: log_duration_pmf(runs[False], round(cfg["alive_max_s"] * STEPS_PER_S),
                                    cfg["bandwidth"])}


def boundary_priors(labelled_steps, sigma_s):
    """
    Training log-odds of the soft start and stop targets.
    """
    sums, count = np.zeros(2), 0
    for dead, period, scored in labelled_steps:
        y, w = targets({"dead": dead, "period": period, "scored": scored}, sigma_s * STEPS_PER_S)
        sums += (y[1:] * w).sum(1)
        count += w.sum()
    return logit(sums / count)


class Decoder:
    def __init__(self, labelled_steps, cfg, sigma_s, temperature=None, w_start=None, w_stop=None):
        """
        cfg is the [decoding] table; temperature, w_start and w_stop override it.
        """
        pick = lambda value, key: cfg[key] if value is None else value  # noqa: E731
        self.setting = (pick(temperature, "temperature"), pick(w_start, "w_start"),
                        pick(w_stop, "w_stop"))
        self.hsmm = HSMM(durations(labelled_steps, cfg), *self.setting)
        self.boundary_prior = boundary_priors(labelled_steps, sigma_s)

    def evidence(self, logits, prior):
        """
        Dead, start and stop evidence per step.
        """
        return (logits[:, 0] - float(prior), logits[:, 1] - self.boundary_prior[0],
                logits[:, 2] - self.boundary_prior[1])

    def decode(self, logits, prior, period):
        """
        Dead mask per step from the heads' logits and the training log-odds of dead.
        """
        dead, start, stop = self.evidence(logits, prior)
        return self.hsmm.decode(dead, period, start, stop)

    def decode_types(self, logits, prior, period):
        """
        Dead mask and class per step. Each predicted stoppage gets the type output's most
        likely class over it, or -1.
        """
        dead = self.decode(logits, prior, period)
        return dead, segment_classes(dead, period, logits)


def segment_classes(dead, period, logits):
    """
    Class per step: each predicted stoppage gets its most likely class on average.
    """
    kind = np.full(len(dead), -1, dtype=np.int64)
    if logits.shape[1] <= 3:
        return kind
    p = type_probabilities(logits)
    change = np.flatnonzero((dead[1:] != dead[:-1]) | (period[1:] != period[:-1])) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, len(dead)]):
        if dead[a]:
            kind[a:b] = int(np.argmax(np.nanmean(p[a:b], axis=0)))
    return kind


def type_probabilities(logits):
    """
    Softmax of the type output, the logits after the first three columns.
    """
    z = logits[:, 3:].astype(float)
    z = z - np.nanmax(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def dead_runs_by_class(dead, period, scored, kind):
    """
    Lengths of complete dead runs and counts of all dead runs, by the class of their first step.
    """
    lengths, counts = {}, {}
    change = np.flatnonzero((dead[1:] != dead[:-1]) | (period[1:] != period[:-1])
                            | (scored[1:] != scored[:-1])) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, len(dead)]):
        if not (dead[a] and scored[a]) or kind[a] < 0:
            continue
        k = int(kind[a])
        counts[k] = counts.get(k, 0) + 1
        if (a > 0 and b < len(dead) and period[a - 1] == period[a] == period[b]
                and scored[a - 1] and scored[b]):
            lengths.setdefault(k, []).append(int(b - a))
    return lengths, counts


class TypedDecoder(Decoder):
    def __init__(self, labelled_steps, step_types, cfg, sigma_s, n_classes, temperature=None,
                 w_start=None, w_stop=None, type_weight=None, type_evidence=None):
        """
        step_types gives, per training match, the class of each step (-1 outside stoppages).
        """
        super().__init__(labelled_steps, cfg, sigma_s, temperature, w_start, w_stop)
        typed = cfg["typed"]
        type_weight = typed["type_weight"] if type_weight is None else type_weight
        type_evidence = typed["evidence"] if type_evidence is None else type_evidence
        self.setting = self.setting + (type_weight, type_evidence)
        lengths, counts = {}, {}
        for (dead, period, scored), kind in zip(labelled_steps, step_types):
            runs, n = dead_runs_by_class(dead, period, scored, kind)
            for k, v in runs.items():
                lengths.setdefault(k, []).extend(v)
            for k, v in n.items():
                counts[k] = counts.get(k, 0) + v
        overall = self.hsmm.log_pmf[DEAD]
        max_steps = round(cfg["dead_max_s"] * STEPS_PER_S)
        pmfs = [log_duration_pmf(lengths[k], max_steps, cfg["bandwidth"]) if lengths.get(k)
                else overall for k in range(n_classes)]
        shares = [max(counts.get(k, 0), 0.5) for k in range(n_classes)]   # an unseen class stays possible
        self.shares = np.asarray(shares) / sum(shares)
        self.typed = TypedHSMM(self.hsmm.log_pmf[ALIVE], pmfs, self.shares, *self.setting[:3],
                               type_weight, type_evidence)

    def type_evidence(self, logits):
        with np.errstate(divide="ignore"):
            return np.log(type_probabilities(logits)) - np.log(self.shares)

    def decode_types(self, logits, prior, period):
        dead, start, stop = self.evidence(logits, prior)
        return self.typed.decode(dead, period, start, stop, self.type_evidence(logits))

    def decode(self, logits, prior, period):
        return self.decode_types(logits, prior, period)[0]


def make_decoder(labelled_steps, step_types, cfg, setting=None):
    """
    Builds the decoder the config asks for, typed or not.
    setting optionally overrides the [decoding] values.
    """
    setting = tuple(setting or ())
    d = cfg["decoding"]
    if d["typed"]["enabled"]:
        return TypedDecoder(labelled_steps, step_types, d, cfg["train"]["boundary_sigma_s"],
                            len(cfg["types"]["classes"]), *setting[:5])
    decoder = Decoder(labelled_steps, d, cfg["train"]["boundary_sigma_s"], *setting[:3])
    decoder.setting = decoder.setting + (0.0, "none")
    return decoder
