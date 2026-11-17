"""
Corrupted copies ("views") of a DFL match, built from the config's recipes, with the TCN's inputs.

A recipe can swap and corrupt the players, move the dead ball, hide the ball and add simulated
whistles. Every draw is seeded by step, match and copy, so views are reproducible.
"""

import numpy as np

from dead_alive_detection import config
from dead_alive_detection.augment.ball.hidden import corrupt_match, hide_ball, load_model
from dead_alive_detection.augment.dead_ball import bar, uniform
from dead_alive_detection.augment.players import fragments, learned, swaps
from dead_alive_detection.augment.whistle import simulated
from dead_alive_detection.evaluation.metrics import pm, truth_labels
from dead_alive_detection.loading.features import step_inputs
from dead_alive_detection.loading.tensors import match_tensors


def inputs(t, cfg):
    """
    The TCN's per-step input for tensors t.
    """
    return step_inputs(t, **cfg["features"])


class Views:
    def __init__(self, cfg):
        self.cfg = cfg
        aug = cfg["augment"]
        used = [config.recipe(cfg, name)
                for name in (aug["train"]["recipe"], *aug["eval"]["conditions"])]
        # Each tool is loaded only if a recipe needs it (the bar library may be built from BAR).
        self.ball_model = load_model(config.resolve_path(aug["ball"]["model"])) \
            if any(r["ball"] == "hidden" for r in used) else None
        kinds = {r["players"] for r in used}
        self.corrupt_players = {}
        if "fragments" in kinds:
            self.corrupt_players["fragments"] = fragments.from_config(aug["players"]["fragments"])
        if "learned" in kinds:
            self.corrupt_players["learned"] = learned.from_config(aug["players"]["learned"])
        self.swap_players = swaps.from_config(aug["swaps"]) \
            if any(r.get("swaps", "none") == "bar" for r in used) else None
        motions = {r["dead_ball"] for r in used}
        self.unfreeze = {}
        if "uniform" in motions:
            self.unfreeze["uniform"] = uniform.from_config(aug["dead_ball"]["uniform"])
        if "bar" in motions:
            self.unfreeze["bar"] = bar.from_config(aug["dead_ball"]["bar"],
                                                   config.resolve_path(cfg["bar"]["match"]))
        whistled = cfg["features"]["whistle"] and any(r.get("whistle", "none") != "none" for r in used)
        self.whistle_durations = simulated.read_durations(
            config.resolve_path(cfg["bar"]["whistles"])) if whistled else None
        self._stoppages = {}
        self._holds = {}

    @staticmethod
    def _rng(section, index, copy, short_on_scored=False):
        key = [section["seed"], index] if short_on_scored and copy == 0 \
            else [section["seed"], index, copy]
        return np.random.default_rng(key)

    def _players(self, match, index, copy, kind):
        """
        The match with its players corrupted by kind, for this copy.
        """
        section = self.cfg["augment"]["players"][kind]
        return self.corrupt_players[kind](match, self._rng(section, index, copy, kind == "fragments"))

    def _swapped(self, match, index, copy, kind):
        """
        The match with its players swapped as the recipe says, unchanged for "none".
        """
        if kind == "none":
            return match
        return self.swap_players(match, self._rng(self.cfg["augment"]["swaps"], index, copy))

    def match_version(self, match, index, copy, recipe, players=None):
        """
        The match after the recipe's swaps, player corruption and dead-ball motion.
        """
        kind = recipe.get("swaps", "none")
        version = self._swapped(match, index, copy, kind)
        if recipe["players"] != "clean":
            version = players(recipe["players"], kind) if players else \
                self._players(version, index, copy, recipe["players"])
        sections = self.cfg["augment"]["dead_ball"]
        if recipe["dead_ball"] == "uniform":
            version = self.unfreeze["uniform"](version, truth_labels(match),
                                               self._rng(sections["uniform"], index, copy, True))
        elif recipe["dead_ball"] == "bar":
            version = self.unfreeze["bar"](version, truth_labels(match),
                                           self._rng(sections["bar"], index, copy))
        return version

    def view(self, match, index, copy, name, hidden_frac=None, players=None):
        """
        Tensors and inputs of recipe name.
        hidden_frac defaults to the share measured on BAR-ATH.
        """
        recipe = config.recipe(self.cfg, name)
        t = match_tensors(self.match_version(match, index, copy, recipe, players))
        if recipe["ball"] == "hidden":
            t = hide_ball(t, self.ball_model, self._rng(self.cfg["augment"]["ball"], index, copy),
                          hidden_frac)
        if recipe.get("keeper", "none") == "hide":
            t = self.hide_keeper_holds(t, match, index, copy)
        if recipe.get("whistle", "none") != "none" and self.whistle_durations is not None:
            t["whistle"] = self.whistles(match, index, copy, recipe["whistle"], t["row"])
        return t, inputs(t, self.cfg)

    def hide_keeper_holds(self, t, match, index, copy):
        """
        Copy of t with the ball hidden while the keeper holds it, as BAR-ATH's tracker often does.
        """
        from dead_alive_detection.augment.ball import keeper_holds
        if match.name not in self._holds:
            self._holds[match.name] = keeper_holds.holds(match)
        section = self.cfg["augment"]["keeper"]
        hidden = keeper_holds.mask(self._holds[match.name], t["row"],
                                   self._rng(section, index, copy), section)
        out = dict(t)
        out["ball_seen"] = t["ball_seen"] & ~hidden
        out["ball"] = np.where(hidden[:, None], 0, t["ball"]).astype(t["ball"].dtype)
        return out

    def stoppages(self, match):
        """
        The true stoppages of a DFL match as (start, stop, class). Needs its events.
        """
        from dead_alive_detection.evaluation.stoppage_types import fine_stoppages
        if match.name not in self._stoppages:
            stops = fine_stoppages(match, pm.label_codes(truth_labels(match)))
            if stops is None:
                raise FileNotFoundError("no DFL events for {}: needed by the whistle recipes"
                                        .format(match.name))
            self._stoppages[match.name] = stops
        return self._stoppages[match.name]

    def whistles(self, match, index, copy, profile, rows):
        """
        Whistle flag per step for this copy, drawn with the given profile.
        """
        section = self.cfg["augment"]["whistle"]
        found = simulated.simulate(self.stoppages(match), match.period, section,
                                   section["profiles"][profile], self.whistle_durations,
                                   self._rng(section, index, copy))
        return simulated.to_steps(found, rows)

    def corrupted_match(self, match, index, copy, name, hidden_frac=None):
        """
        The corrupted 25 fps match and ball-hiding report of recipe name, with the same draws as view().
        """
        recipe = config.recipe(self.cfg, name)
        version = self.match_version(match, index, copy, recipe)
        if recipe["ball"] != "hidden":
            return version, {}
        return corrupt_match(version, self.ball_model,
                             self._rng(self.cfg["augment"]["ball"], index, copy), hidden_frac)

    def training(self, match, index):
        """
        The training copies of a match, one per hidden_frac.
        """
        train = self.cfg["augment"]["train"]
        return [self.view(match, index, 1 + k, train["recipe"], frac)
                for k, frac in enumerate(train["hidden_fracs"])]

    def conditions(self, match, index, names=None):
        """
        The scored views of a held-out match, per condition.
        """
        cache = {}

        def players(kind, swapped="none"):
            if (kind, swapped) not in cache:
                cache[kind, swapped] = self._players(self._swapped(match, index, 0, swapped),
                                                     index, 0, kind)
            return cache[kind, swapped]
        return {name: self.view(match, index, 0, name, None, players)
                for name in names or self.cfg["augment"]["eval"]["conditions"]}
