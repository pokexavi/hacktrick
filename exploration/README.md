# exploration/

What is left of the team's exploration toolkit is what the final pipeline still uses:

| path | what | used by |
|---|---|---|
| [common.py](common.py) | parses the competition's ATD / Smart Tagging / homography by **video frame**, cached as parquet under `cache/parsed/` | `external/convert_atd.py` |
| [synthetic/](synthetic/) | made-up data in exactly the format of the competition bundle (no match data), and its generator | the tests, and any dry run of the pipeline without the real data |

The exploration notebooks, the clip renderer and the format probe are not part of this
repository.

A full-length synthetic bundle (same frame count as the real match) can be generated with
`python exploration/synthetic/make_synthetic.py --full --out cache/synthetic_full` and selected with
`MTAH_DATA_DIR=cache/synthetic_full`.
