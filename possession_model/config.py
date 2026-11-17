"""
Loads configs/v1_frozen.toml, applies the --set changes and saves the config of each run.
An option that isn't in the toml is an error, so a typo can't silently do nothing.
"""

import copy
import json
import os

try:
    import tomllib
except ImportError:                      # Python < 3.11 (some Kaggle images): pip install tomli
    import tomli as tomllib

PACKAGE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE)
CONFIG = os.path.join(PACKAGE, "configs", "v1_frozen.toml")

# Versions of a DFL match: (how players are lost, is the ball hidden?). The dead ball is always DFL's.
RECIPES = {"clean": ("clean", False),
           "bar_motion": ("fragments", True),
           "bar_motion_crowd": ("learned", True)}
TRAIN_RECIPE = "bar_motion_crowd"                       # what every training copy uses
CONDITIONS = ("clean", "bar_motion", "bar_motion_crowd")  # what every held-out match is scored on
PRIMARY = "bar_motion"                                  # the one we look at


def _read(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def merge(base, top):
    # Nested tables merge, everything else is replaced
    out = copy.deepcopy(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text):
    # "section.key=value", value in TOML syntax (a bare word is read as a string)
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


def _check_keys(cfg, defaults, where=()):
    for key, value in cfg.items():
        here = where + (key,)
        if key not in defaults:
            raise KeyError("unknown config key {} (not in configs/v1_frozen.toml)".format(".".join(here)))
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise TypeError("config key {} must be a table".format(".".join(here)))
            _check_keys(value, defaults[key], here)


def _check(cfg):
    enc, tem = cfg["model"]["encoder"], cfg["model"]["temporal"]
    if enc["d"] % enc["heads"] or enc["d"] % tem["heads"]:
        raise ValueError("model: d must be divisible by the number of heads")
    if not 0 <= enc["biased_heads"] <= enc["heads"]:
        raise ValueError("model.encoder.biased_heads must be between 0 and heads")
    if 25 % cfg["steps"]["hz"]:
        raise ValueError("steps.hz must divide 25 (1, 5 or 25)")


def load(path=None, overrides=None):
    # The toml (or a run's saved config.toml), then the --set changes
    defaults = _read(CONFIG)
    raw = _read(path) if path else {}
    cfg = apply_overrides(merge(defaults, raw), overrides)
    _check_keys(cfg, defaults)
    _check(cfg)
    return cfg


def resolve_path(path):
    # Paths in the config are relative to the repo root
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


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
        handle.write("# Resolved config (every option): written by the run.\n")
        handle.write(dumps(cfg))
