"""
Averages the saved heads of several runs into one heads folder.

    python -m dead_alive_detection.decoding.ensemble --out <folder> --members <folder> <folder> ...

Probabilities are averaged rather than logits, so one overconfident member cannot outvote the
others. The result is an ordinary heads folder that the grid, explain and predict can use.
"""

import argparse
import json
import os

import numpy as np

EPS = 1e-6
SAME = ("period", "dead", "scored")     # must agree between members, step by step


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nearest(rows, period, member, tolerance):
    """
    Index of the member's step nearest each row, and whether it is close enough in the same half.
    """
    idx = np.clip(np.searchsorted(member["row"], rows), 1, len(member["row"]) - 1)
    left = np.abs(member["row"][idx - 1] - rows) <= np.abs(member["row"][idx] - rows)
    idx = np.where(left, idx - 1, idx)
    ok = (np.abs(member["row"][idx] - rows) <= tolerance) & (member["period"][idx] == period)
    return idx, ok


def combine(parts):
    """
    Combines the members' versions of one heads file into one.
    Steps are matched to the first member's by nearest frame, since older runs step BAR on
    another grid.
    """
    first = parts[0]
    tolerance = int(np.median(np.diff(first["row"]))) // 2 if len(first["row"]) > 1 else 0
    keep = np.ones(len(first["row"]), bool)
    matched = []
    for p in parts:
        idx, ok = _nearest(first["row"], first["period"], p, tolerance)
        matched.append(idx)
        keep &= ok
    taken = [idx[keep] for idx in matched]
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == first["row"].shape else v)
           for k, v in first.items()}
    for key in SAME:
        for p, idx in zip(parts[1:], taken[1:]):
            if not np.array_equal(p[key][idx], out[key]):
                raise ValueError("members disagree on '{}': not the same matches or folds".format(key))
    priors = {round(float(p["prior"]), 9) for p in parts}
    if len(priors) > 1:
        raise ValueError("members have different training priors: {}".format(sorted(priors)))
    heads = [p["logits"][idx] for p, idx in zip(parts, taken)]
    logits = _logit(np.mean([1 / (1 + np.exp(-h[:, :3].astype(float))) for h in heads], axis=0))
    widths = {h.shape[1] for h in heads}
    if len(widths) == 1 and widths != {3}:
        mean = np.mean([_softmax(h[:, 3:].astype(float)) for h in heads], axis=0)
        logits = np.column_stack([logits, np.log(np.clip(mean, EPS, None))])
    elif "type" in out:
        del out["type"]
    out["logits"] = logits.astype(np.float32)
    return out


def write(out, members):
    """
    Writes the ensemble of every heads file all members have, and returns their names.
    """
    names = sorted(set.intersection(*[{f for f in os.listdir(m) if f.endswith(".npz")
                                       and (f.startswith("oof_") or f == "bar.npz")}
                                      for m in members]))
    if not names:
        raise ValueError("the members have no heads file in common")
    os.makedirs(out, exist_ok=True)
    for name in names:
        parts = [dict(np.load(os.path.join(m, name))) for m in members]
        np.savez_compressed(os.path.join(out, name), **combine(parts))
    with open(os.path.join(out, "members.json"), "w") as handle:
        json.dump({"members": [os.path.normpath(m).replace("\\", "/") for m in members],
                   "how": "mean of probabilities (dead, start, stop; types when all members have them)",
                   "files": names}, handle, indent=1)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="the ensemble's heads folder")
    parser.add_argument("--members", nargs="+", required=True, help="heads folders to average")
    args = parser.parse_args(argv)
    if len(args.members) < 2:
        parser.error("an ensemble needs at least two members")
    names = write(args.out, args.members)
    print("{}: {} files from {} members".format(args.out, len(names), len(args.members)))


if __name__ == "__main__":
    main()
