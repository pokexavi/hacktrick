"""
Makes DFL look like BAR: loses players and hides the ball the way BAR's tracking does.
Uses the same corruptions as the dead-ball model. The dead ball is left as in DFL.
"""

from dataclasses import replace

import numpy as np

from possession_model import config


def true_live(label):
    return np.isin(label, ("H", "A"))


class Corruptor:
    def __init__(self, cfg):
        self.cfg = cfg
        self._tools = {}

    def _tool(self, kind):
        # The BAR-fitted models are only loaded the first time a recipe needs them
        if kind not in self._tools:
            aug = self.cfg["augment"]
            if kind == "fragments":
                from dead_alive_detection.augment.players import fragments
                self._tools[kind] = fragments.from_config(aug["fragments"])
            elif kind == "learned":
                from dead_alive_detection.augment.players import learned
                self._tools[kind] = learned.from_config(aug["learned"])
            else:
                from dead_alive_detection.augment.ball.hidden import load_model
                self._tools[kind] = load_model(config.resolve_path(aug["ball"]["model"]))
        return self._tools[kind]

    @staticmethod
    def _rng(section, index, copy):
        return np.random.default_rng([section["seed"], index, copy])

    def corrupt(self, m, recipe, copy=0, hidden_frac=None):
        """
        Returns the corrupted match and its live mask. copy is 0 for a scored match and 1 + k for
        training copy k; hidden_frac is the share of hidden ball (None = BAR's).
        """
        players, hide = config.RECIPES[recipe]
        aug = self.cfg["augment"]
        version = m.match
        if players != "clean":
            version = self._tool(players)(version, self._rng(aug[players], m.index, copy))
        if hide:
            from dead_alive_detection.augment.ball.hidden import corrupt_match
            version, _ = corrupt_match(version, self._tool("ball"), self._rng(aug["ball"], m.index, copy),
                                       hidden_frac)
        live = true_live(m.match.label) & (m.match.period > 0)
        # Blank the labels so the model can never read them from its input
        return replace(version, label=np.full(len(version.label), "X", dtype="<U1")), live
