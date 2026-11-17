"""
Reads and writes the annotation logs (labels/<user>.jsonl) and builds the labels from them.
The possession model uses it when training: read_logs, model_labels and labels_fingerprint.
"""

import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from external import schema  # noqa: E402

# Params
DATASETS_DIR = os.environ.get("POSSESSION_DATASETS_DIR") or os.path.join(HERE, "datasets")
LABELS_DIR = os.environ.get("POSSESSION_LABELS_DIR") or os.path.join(HERE, "labels")
SOURCES = ("dfl", "metrica")
VERDICTS = ("ok", "edited", "unsure")
KEEP_VERDICTS = ("ok", "edited")      # a reviewed transition survives the merge
KINDS = ("H/A", "to D", "from D", "X")

# One JSON per line. An edit relabels frames [start, stop); a review marks one transition as seen.
EDIT_FIELDS = ("user", "source", "match", "start", "stop", "label")
REVIEW_FIELDS = ("user", "source", "match", "frame", "from", "to", "verdict")


# Paths

def original_path(source, match, root=None):
    return os.path.join(root or DATASETS_DIR, source, "original", match + ".parquet")


def working_path(source, match, root=None):
    return os.path.join(root or DATASETS_DIR, source, "working", match + ".parquet")


def matches(source, root=None):
    folder = os.path.join(root or DATASETS_DIR, source, "original")
    if not os.path.isdir(folder):
        return []
    return sorted(name[:-len(".parquet")] for name in os.listdir(folder) if name.endswith(".parquet"))


def safe_user(user):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", user.strip()).strip("._")
    if not name:
        raise ValueError("empty user name")
    return name


# Logs

def read_logs(labels_dir=None):
    """
    Every record of every log, in order (files by name, then line by line), plus the bad lines.
    """
    labels_dir = labels_dir or LABELS_DIR
    records, problems = [], []
    if not os.path.isdir(labels_dir):
        return records, problems
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(labels_dir, name), encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    _check(record)
                except (ValueError, KeyError, TypeError) as error:
                    problems.append("{}:{}: {}".format(name, line_no, error))
                    continue
                record.pop("ts", None)          # old logs had a timestamp, we no longer use it
                record["_order"] = (name, line_no)
                records.append(record)
    records.sort(key=lambda r: r["_order"])
    return records, problems


def _check(record):
    kind = record.get("kind")
    if kind == "edit":
        missing = [f for f in EDIT_FIELDS if f not in record]
        if missing:
            raise ValueError("edit without {}".format(missing))
        if record["label"] not in schema.LABELS:
            raise ValueError("label {!r} not in {}".format(record["label"], schema.LABELS))
        if int(record["stop"]) <= int(record["start"]):
            raise ValueError("empty edit range")
    elif kind == "review":
        missing = [f for f in REVIEW_FIELDS if f not in record]
        if missing:
            raise ValueError("review without {}".format(missing))
        if record["verdict"] not in VERDICTS:
            raise ValueError("verdict {!r} not in {}".format(record["verdict"], VERDICTS))
    else:
        raise ValueError("unknown kind {!r}".format(kind))


def append(user, records, labels_dir=None):
    labels_dir = labels_dir or LABELS_DIR
    os.makedirs(labels_dir, exist_ok=True)
    stamped = []
    for record in records:
        record = dict(record, user=user)
        _check(record)
        stamped.append(record)
    with open(os.path.join(labels_dir, safe_user(user) + ".jsonl"), "a", encoding="utf-8") as handle:
        for record in stamped:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return stamped


def for_match(records, source, match, kind):
    return [r for r in records if r["kind"] == kind and r["source"] == source and r["match"] == match]


# Labels

def apply_edits(label, frame, period, edits):
    # Later edits win. Frames outside the two halves are never touched.
    label = np.array(label, dtype="<U1", copy=True)
    frame = np.asarray(frame)
    in_halves = np.asarray(period) > 0
    for edit in edits:
        rows = (frame >= int(edit["start"])) & (frame < int(edit["stop"])) & in_halves
        label[rows] = edit["label"]
    return label


def label_runs(label):
    # [[start, stop, label]] with row indices, stop excluded
    label = np.asarray(label)
    if not len(label):
        return []
    edges = np.r_[0, np.flatnonzero(label[1:] != label[:-1]) + 1, len(label)]
    return [[int(a), int(b), str(label[a])] for a, b in zip(edges[:-1], edges[1:])]


def labels_from_runs(runs, length):
    # The opposite of label_runs. The runs have to cover [0, length) with no gaps.
    out = np.full(length, "X", dtype="<U1")
    position = 0
    for start, stop, value in runs:
        if start != position or stop <= start or value not in schema.LABELS:
            raise ValueError("runs do not tile the window: {}".format(runs))
        out[start:stop] = value
        position = stop
    if position != length:
        raise ValueError("runs end at {} instead of {}".format(position, length))
    return out


def diff_edits(old, new, frame, offset=0):
    # The edits that turn old into new. Both cover the rows from offset onwards.
    old, new = np.asarray(old), np.asarray(new)
    edits = []
    for start, stop, value in label_runs(np.where(old != new, new, "")):
        if value:
            edits.append({"start": int(frame[offset + start]), "stop": end_frame(frame, offset + stop),
                          "label": value})
    return edits


def kind_of(before, after):
    if "X" in (before, after):
        return "X"
    if after == "D":
        return "to D"
    if before == "D":
        return "from D"
    return "H/A"


def transitions(label, period, frame, fps=schema.FPS):
    """
    One row per boundary between two spells of the same half: where it is, the labels on each
    side and how long those two spells last.
    """
    columns = ["row", "frame", "period", "from", "to", "kind", "prev_row", "next_row",
               "before_s", "after_s"]
    spells = schema.spells(label, period)
    period = np.asarray(period)
    rows = []
    for (a, b, left), (c, d, right) in zip(spells[:-1], spells[1:]):
        if period[a] != period[c] or period[c] == 0:
            continue
        rows.append((c, int(frame[c]), int(period[c]), left, right, kind_of(left, right), a, d,
                     (b - a) / fps, (d - c) / fps))
    return pd.DataFrame(rows, columns=columns)


def merge_removes(table, label, period, min_frames):
    # True for the transitions the merge would erase (a short spell between two of the other team).
    from external.labels import controlled
    merged = controlled(np.asarray(label), np.asarray(period), int(min_frames))
    rows = table["row"].to_numpy()
    return (merged[rows - 1] == merged[rows]) & (np.asarray(label)[rows - 1] != np.asarray(label)[rows])


def model_labels(label, frame, period, records, source, match, min_frames):
    """
    The labels the possession model trains on: DFL's labels + the edits, then the merge, except
    around the transitions someone marked ok or edited.
    """
    from external.labels import controlled
    period = np.asarray(period)
    edited = apply_edits(label, frame, period, for_match(records, source, match, "edit"))
    merged = controlled(edited, period, int(min_frames))
    reviews = for_match(records, source, match, "review")
    if reviews:
        table = review_status(transitions(edited, period, frame), reviews)
        for t in table[table.last_verdict.isin(KEEP_VERDICTS)].itertuples():
            merged[t.prev_row:t.next_row] = edited[t.prev_row:t.next_row]
    return merged


def labels_fingerprint(records, source):
    # Hash of the logs, so we know which annotations a model was trained with. Order matters.
    payload = json.dumps([json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, sort_keys=True)
                          for r in records if r["source"] == source])
    return hashlib.sha1(payload.encode()).hexdigest()


def merge_min_frames(default=25):
    # The min_frames the possession model uses, so the app shows the same merge
    try:
        from possession_model import config
        return int(config.load()["data"]["min_frames"])
    except (ImportError, OSError, KeyError, ValueError):
        return default


def review_status(table, reviews):
    # Adds who reviewed each transition and the latest verdict. The last review of each user counts.
    latest = {}
    for review in reviews:
        key = (int(review["frame"]), review["from"], review["to"])
        users = latest.setdefault(key, {})
        users.pop(review["user"], None)      # re-insert so the dict keeps the order of the latest reviews
        users[review["user"]] = review
    reviewed_by, last_user, last_verdict, last_note = [], [], [], []
    for row in table.to_dict("records"):
        found = list(latest.get((int(row["frame"]), row["from"], row["to"]), {}).values())
        reviewed_by.append([(r["user"], r["verdict"], r.get("note", "")) for r in found])
        last = found[-1] if found else None
        last_user.append(last["user"] if last else None)
        last_verdict.append(last["verdict"] if last else None)
        last_note.append(last.get("note", "") if last else None)
    table = table.copy()
    table["reviewed_by"] = reviewed_by
    table["last_user"], table["last_verdict"], table["last_note"] = last_user, last_verdict, last_note
    table["reviewed"] = table["last_user"].notna()
    return table


# Working copies (original + edits) for the app

def edits_hash(edits):
    payload = json.dumps([[e["user"], int(e["start"]), int(e["stop"]), e["label"]] for e in edits])
    return hashlib.sha1(payload.encode()).hexdigest()


def read_original(source, match, root=None):
    path = original_path(source, match, root)
    # Same encoding as schema.write_match / load_match (the sidecar has team names)
    with open(os.path.splitext(path)[0] + ".json") as handle:
        meta = json.load(handle)
    return pd.read_parquet(path), meta


def rebuild(source, match, records, root=None, force=False):
    # Rewrites working/<match>.parquet only if its edits changed. Returns (path, rebuilt).
    edits = for_match(records, source, match, "edit")
    digest = edits_hash(edits)
    path = working_path(source, match, root)
    sidecar = os.path.splitext(path)[0] + ".json"
    if not force and os.path.isfile(path) and os.path.isfile(sidecar):
        with open(sidecar) as handle:
            if json.load(handle).get("labels_hash") == digest:
                return path, False
    table, meta = read_original(source, match, root)
    table["label"] = apply_edits(table["label"].to_numpy(), table["frame"].to_numpy(),
                                 table["period"].to_numpy(), edits)
    meta = {k: v for k, v in meta.items() if k != "schema_version"}
    meta.update(labels_hash=digest, n_edits=len(edits),
                editors=sorted({e["user"] for e in edits}), original_labels=meta.get("labels"))
    schema.write_match(table, meta, os.path.dirname(path), match)
    return path, True


def rebuild_all(records, root=None):
    done = []
    for source in SOURCES:
        for match in matches(source, root):
            path, rebuilt = rebuild(source, match, records, root)
            if rebuilt:
                done.append(path)
    return done


# Edits for the app's buttons

def end_frame(frame, row):
    # Frame id where a range ending at `row` stops (one past the last frame at the end of the table)
    return int(frame[row]) if row < len(frame) else int(frame[-1]) + 1


def move_boundary(frame, transition, new_row, left, right):
    a, d = int(transition["prev_row"]), int(transition["next_row"])
    if not a <= new_row <= d:
        raise ValueError("new boundary outside the two spells it separates")
    edits = []
    if new_row > a:
        edits.append({"start": int(frame[a]), "stop": end_frame(frame, new_row), "label": left})
    if new_row < d:
        edits.append({"start": int(frame[new_row]), "stop": end_frame(frame, d), "label": right})
    return edits


def merge(frame, transition, into):
    # "previous": the spell after takes the label before. "next": the other way round.
    a, c, d = int(transition["prev_row"]), int(transition["row"]), int(transition["next_row"])
    if into == "previous":
        return [{"start": int(frame[c]), "stop": end_frame(frame, d), "label": transition["from"]}]
    return [{"start": int(frame[a]), "stop": end_frame(frame, c), "label": transition["to"]}]
