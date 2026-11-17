"""Pure parsers for the browser viewer. No filesystem or data loaders."""
import csv
import io
import json
import math
import xml.etree.ElementTree as ET


def parse_labels(text):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    names = {name.strip().lower(): name for name in (reader.fieldnames or [])}
    if not {"frame", "label"} <= names.keys():
        raise ValueError("CSV: expected frame,label columns")
    result = {}
    for row in reader:
        frame = int(row[names["frame"]])
        label = row[names["label"]].strip().upper()
        if frame < 0 or frame in result or label not in "HADX" or len(label) != 1:
            raise ValueError("CSV: duplicate/negative frame or label outside H/A/D/X")
        result[frame] = label
    if not result:
        raise ValueError("Empty CSV")
    return result


def segments(labels):
    out = []
    for frame, label in sorted(labels.items()):
        if out and out[-1][1] == frame and out[-1][2] == label:
            out[-1][1] = frame + 1
        else:
            out.append([frame, frame + 1, label])
    return out


def parse_events(text, fps):
    events = []
    for instance in ET.fromstring(text).iter("instance"):
        start = float(instance.findtext("start"))
        end = float(instance.findtext("end"))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("XML: invalid time interval")
        events.append(dict(start=round(start * fps), end=round(end * fps),
                           code=instance.findtext("code") or "Unknown code",
                           tags={label.findtext("group") or "Tag": label.findtext("text") or ""
                                 for label in instance.findall("label")}))
    return sorted(events, key=lambda event: event["start"])


def payload_json(payload):
    # An uploaded name/tag must never terminate the script element.
    return json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")


def parse_atd(lines, metadata_text, start, end):
    """Parse a window from FIFA/EPTS raw lines and metadata into a pitch view.

    Slots follow XML Player order. Positions remain in normalized field coordinates;
    non-finite positions are absent. ATD frame_id is one-based, video frame is zero-based.
    """
    root = ET.fromstring(metadata_text)
    teams = list(root.iter("Team"))
    players = list(root.iter("Player"))
    if len(teams) < 2 or not players:
        raise ValueError("ATD XML must define two teams and player slots.")
    sides = {teams[0].get("id"): "H", teams[1].get("id"): "A"}
    slots = [{"id": player.get("id") or str(i + 1),
              "team": sides.get(player.get("teamId"), "X")}
             for i, player in enumerate(players)]
    frames = {}
    previous = -1

    def position(value):
        fields = value.split(",")
        if len(fields) != 3:
            raise ValueError("ATD objects must contain x,y,s.")
        x, y, speed = map(float, fields)
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return [x, y, speed if math.isfinite(speed) else None]

    for line in lines:
        if not line.strip():
            continue
        fields = line.strip().split(":")
        if len(fields) != 3:
            raise ValueError("ATD lines must contain frame_id:players:ball.")
        frame = int(fields[0]) - 1
        if frame < 0 or frame <= previous:
            raise ValueError("ATD frame IDs must be positive and strictly increasing.")
        previous = frame
        if frame >= end:
            break
        if frame < start:
            continue
        values = fields[1].split(";")
        if len(values) != len(slots):
            raise ValueError("ATD TXT slot count does not match the metadata XML.")
        frames[frame] = {"players": [position(value) for value in values],
                         "ball": position(fields[2])}
    if not frames:
        raise ValueError("ATD contains no frames in the selected window.")
    return {"slots": slots, "frames": frames}
