"""
Loads, checks and saves experiment configs.

A config is a TOML file that can start with `extends = "base.toml"` and only list what it changes.
base.toml has every option, so an unknown key is an error. `--set section.key=value` overrides
one value, and configs in older formats are translated by migrate().
"""

import copy
import json
import os
import tomllib

PACKAGE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(PACKAGE, "configs")
BASE = os.path.join(CONFIGS, "base.toml")
DATASETS = ("dfl", "metrica")                 # the sources of external/dfl/processed
OPEN_TABLES = (("augment", "recipes"), ("augment", "whistle", "profiles"))   # keys are names
CHOICES = {"players": ("clean", "fragments", "learned"),       # the first choice is the default
           "ball": ("clean", "hidden"),
           "dead_ball": ("frozen", "uniform", "bar")}
WHISTLE_PROFILE = ("recall", "false_per_min")   # recipe key whistle: "none" or a profile name
SWAPS = ("none", "bar")       # optional recipe key swaps (absent = "none", so old recipes stay equal)
KEEPER = ("none", "hide")     # optional recipe key keeper (absent = "none", so old recipes stay equal)


def _read(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def merge(base, top):
    """
    Lays top over base: tables merge recursively, other values replace.
    """
    out = copy.deepcopy(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(path, seen=()):
    """
    Raw TOML of path with its extends chain merged in, without defaults or checks.
    """
    path = os.path.abspath(path)
    if path in seen:
        raise ValueError("config extends itself: " + " -> ".join(seen + (path,)))
    raw = _read(path)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return merge(_resolve(os.path.join(os.path.dirname(path), parent), seen + (path,)), raw)


def _migrate_first(raw):
    """
    Converts a first-format config (a [ball_loss] section, no [augment]) to use [augment].
    """
    if "augment" in raw or "ball_loss" not in raw and "decoding" not in raw:
        return raw
    raw = copy.deepcopy(raw)
    for switch in ("ball", "collective", "last_seen"):  # absent meant off in that format
        raw.setdefault("features", {}).setdefault(switch, False)
    for section, key in (("model", "type"), ("decoding", "method"), ("train", "threshold")):
        raw.get(section, {}).pop(key, None)            # obsolete: one model, one decoder
    loss = raw.pop("ball_loss", None)
    augment = {"train": {}, "eval": {}}
    if loss is None:                                   # nothing hidden anywhere
        augment["train"] = {"recipe": "clean", "hidden_fracs": [0.0]}
        augment["eval"] = {"conditions": ["clean"], "primary": "clean"}
    else:
        augment["ball_loss_model"] = loss["model"]
        augment["seed"] = loss["seed"]
        augment["train"] = {"recipe": "bar_ball" if loss["train"] else "clean",
                            "hidden_fracs": loss["train_fracs"] if loss["train"] else [0.0]}
        scored = "bar_ball" if loss["eval"] else "clean"
        augment["eval"] = {"conditions": [scored], "primary": scored}
    raw["augment"] = augment
    return raw


def _migrate_recipe(old):
    new = dict(old)
    if new.get("players") == "bar":
        new["players"] = "fragments"
    if "ball_motion" in new:
        new["dead_ball"] = new.pop("ball_motion")
    if "hide_ball" in new:
        new["ball"] = "hidden" if new.pop("hide_ball") else "clean"
    return new


def _migrate_augment(raw):
    """
    Updates an older [augment] table and its recipe keys to the current names.
    """
    if "augment" not in raw:
        return raw
    raw = copy.deepcopy(raw)
    aug = raw["augment"]
    if "ball_loss_model" in aug:
        aug.setdefault("ball", {})["model"] = aug.pop("ball_loss_model")
    if "seed" in aug:
        aug.setdefault("ball", {})["seed"] = aug.pop("seed")
    players = aug.get("players", {})
    flat = {k: players.pop(k) for k in list(players) if k not in CHOICES["players"]}
    if flat:
        players["fragments"] = flat
    for old, new in (("ball_motion", "uniform"), ("bar_motion", "bar")):
        if old in aug:
            aug.setdefault("dead_ball", {})[new] = aug.pop(old)
    if "recipes" in aug:
        aug["recipes"] = {name: _migrate_recipe(r) for name, r in aug["recipes"].items()}
    return raw


def migrate(raw):
    """
    Converts a config in any older format to the current one.
    """
    return _migrate_augment(_migrate_first(raw))


def _check(cfg, defaults, where=()):
    for key, value in cfg.items():
        here = where + (key,)
        if key not in defaults:
            raise KeyError("unknown config key {} (not in base.toml)".format(".".join(here)))
        if isinstance(defaults[key], dict) and here not in OPEN_TABLES:
            if not isinstance(value, dict):
                raise TypeError("config key {} must be a table".format(".".join(here)))
            _check(value, defaults[key], here)


def _check_datasets(cfg):
    unknown = set(cfg["data"]["datasets"]) - set(DATASETS)
    if unknown:
        raise ValueError("data.datasets: unknown {}, known {}".format(sorted(unknown), DATASETS))


def _check_recipes(cfg):
    recipes = cfg["augment"]["recipes"]
    profiles = cfg["augment"]["whistle"]["profiles"]
    for name, profile in profiles.items():
        if sorted(profile) != sorted(WHISTLE_PROFILE):
            raise KeyError("whistle profile {} must have exactly {}".format(name, WHISTLE_PROFILE))
    for name, recipe in recipes.items():
        unknown = set(recipe) - set(CHOICES) - {"whistle", "swaps", "keeper"}
        if unknown:
            raise KeyError("recipe {}: unknown keys {}".format(name, sorted(unknown)))
        if recipe.get("swaps", "none") not in SWAPS:
            raise ValueError("recipe {}: swaps must be one of {}".format(name, SWAPS))
        if recipe.get("keeper", "none") not in KEEPER:
            raise ValueError("recipe {}: keeper must be one of {}".format(name, KEEPER))
        for key, allowed in CHOICES.items():
            if recipe.get(key, allowed[0]) not in allowed:
                raise ValueError("recipe {}: {} must be one of {}".format(name, key, allowed))
        if recipe.get("whistle", "none") not in ("none", *profiles):
            raise ValueError("recipe {}: whistle must be none or a profile of [augment.whistle."
                             "profiles] {}".format(name, sorted(profiles)))
    used = [cfg["augment"]["train"]["recipe"], *cfg["augment"]["eval"]["conditions"]]
    missing = [name for name in used if name not in recipes]
    if missing:
        raise KeyError("recipes used but not defined in [augment.recipes]: {}".format(missing))
    if cfg["augment"]["eval"]["primary"] not in cfg["augment"]["eval"]["conditions"]:
        raise ValueError("augment.eval.primary must be one of augment.eval.conditions")


def _check_feature_set(cfg):
    if cfg["features"]["feature_set"] not in ("v1", "v2"):
        raise ValueError('features.feature_set must be "v1" or "v2"')


def _check_types(cfg):
    from dead_alive_detection.evaluation.stoppage_types import FINE_CLASSES
    classes = cfg["types"]["classes"]
    unknown = [c for c in classes if c not in FINE_CLASSES]
    if unknown or len(set(classes)) != len(classes):
        raise ValueError("types.classes: unknown or repeated {} (known: {})".format(unknown, FINE_CLASSES))
    if classes and "other" not in classes:
        raise ValueError('types.classes must include "other" (stoppages no class explains)')
    if cfg["decoding"]["typed"]["enabled"] and not classes:
        raise ValueError("decoding.typed needs types.classes (the TCN's type output)")
    if cfg["decoding"]["typed"]["evidence"] not in ("mean", "sum"):
        raise ValueError('decoding.typed.evidence must be "mean" or "sum"')


def recipe(cfg, name):
    """
    Recipe `name` with its choice keys filled with defaults.
    Optional keys like whistle and swaps are only present if the recipe sets them.
    """
    return {**{key: allowed[0] for key, allowed in CHOICES.items()},
            **cfg["augment"]["recipes"][name]}


def parse_override(text):
    """
    Splits "section.key=value" into (key, value), parsing value as TOML or else as a string.
    """
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError("--set expects section.key=value, got " + text)
    try:
        parsed = tomllib.loads("v = " + value)["v"]
    except tomllib.TOMLDecodeError:
        parsed = value
    return key.strip(), parsed


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for text in overrides or ():
        key, value = parse_override(text)
        *path, last = key.split(".")
        table = cfg
        for part in path:
            table = table.setdefault(part, {})
        table[last] = value
    return cfg


def load(path, overrides=None):
    """
    Full checked config: base.toml defaults, the file's extends chain, then the overrides.
    """
    defaults = _read(BASE)
    raw = migrate(_resolve(path))
    cfg = apply_overrides(merge(defaults, raw), overrides)
    _check(cfg, defaults)
    _check_datasets(cfg)
    _check_recipes(cfg)
    _check_types(cfg)
    _check_feature_set(cfg)
    if not cfg["name"]:
        cfg["name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


def resolve_path(path):
    """
    Makes a path relative to the repository root absolute.
    """
    root = os.path.dirname(PACKAGE)
    return path if os.path.isabs(path) else os.path.join(root, path)


# --- writing (tomllib reads only) --------------------------------------------------------------

def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join("{} = {}".format(k, _value(x)) for k, x in v.items()) + "}"
    raise TypeError("cannot write {!r} to TOML".format(v))


def dumps(cfg):
    lines = []

    def table(d, where):
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        if where and (scalars or not d):
            lines.append("\n[{}]".format(".".join(where)))
        for k, v in scalars.items():
            lines.append("{} = {}".format(k, _value(v)))
        for k, v in d.items():
            if isinstance(v, dict):
                table(v, where + (k,))

    table(cfg, ())
    return "\n".join(lines).lstrip("\n") + "\n"


def dump(cfg, path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Resolved config (every option, no extends): written by the run.\n")
        handle.write(dumps(cfg))
