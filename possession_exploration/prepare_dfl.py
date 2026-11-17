"""
Copies the 7 DFL matches from external/dfl/processed into datasets/dfl/original for the app.
Use --force to copy them again. If there is nothing to copy, run external/fetch_dfl.py first.
"""

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from external import schema  # noqa: E402
from possession_exploration import labelstore  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--force", action="store_true", help="replace existing copies")
    args = parser.parse_args(argv)
    sources = sorted(glob.glob(os.path.join(schema.DFL_DIR, "processed", "DFL-MAT-*.parquet")))
    if not sources:
        sys.exit("no processed DFL matches in {} -- run python external/fetch_dfl.py first".format(
            os.path.join(schema.DFL_DIR, "processed")))
    out = os.path.dirname(labelstore.original_path("dfl", "x"))
    os.makedirs(out, exist_ok=True)
    for parquet in sources:
        stem = os.path.splitext(os.path.basename(parquet))[0]
        target = os.path.join(out, stem + ".parquet")
        if os.path.isfile(target) and not args.force:
            print("  {} already there".format(stem))
            continue
        shutil.copyfile(os.path.splitext(parquet)[0] + ".json", os.path.join(out, stem + ".json"))
        shutil.copyfile(parquet, target)
        print("  {} copied".format(stem))
    records, _ = labelstore.read_logs()
    rebuilt = [p for m in labelstore.matches("dfl")
               for p, done in [labelstore.rebuild("dfl", m, records, force=args.force)] if done]
    print("{} DFL matches in {}, {} working copies rebuilt".format(
        len(labelstore.matches("dfl")), out, len(rebuilt)))


if __name__ == "__main__":
    main()
