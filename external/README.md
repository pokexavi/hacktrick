# External data: Sportec Open DFL

Seven professional matches of tracking data with **per-frame possession and ball-status flags**.
That is the same kind of label our ground truth is built from, and seven matches is ~11 hours of tracking (1M frames),
against our 5 labelled minutes. This folder downloads the matches, converts them and our BAR-ATH
tracking into **one shared format**, so a model never needs to know which source it is reading.

## Setup (once per teammate)

```bash
pip install -r requirements-pipeline.txt       # from the repository root
python external/fetch_dfl.py                   # ~2.6 GB download, ~1 min of conversion
python external/convert_atd.py                 # our match -> data/processed/BAR-ATH.parquet  (needs data/)
python external/noise.py                       # BAR-ATH tracking-noise aggregates -> data/processed/noise_profile.json
```

`fetch_dfl.py` is safe to re-run: verified files are skipped and interrupted downloads resume. Use
`--matches J03WOH` for a quick one-match test and `--delete-raw` to drop the 2.6 GB of XML after
conversion.

| where | what | in git? |
|---|---|---|
| `external/dfl/raw/` | original DFL XMLs, MD5-checked against figshare | no |
| `external/dfl/processed/` | one parquet + JSON per DFL match | no |
| `data/processed/` | BAR-ATH in the same format | no (competition data) |

## The shared format ([schema.py](schema.py))

One row per frame at 25 fps: `frame, period, ball_x/y/z, h1_x, h1_y, ..., a1_x, ..., label`.

- **Metres, origin at the centre spot. Home always attacks +x** (each half is rotated). For DFL this is
  checked from the kickoff positions and the goalkeepers. For BAR-ATH it comes from the metadata and is
  cross-checked at both kickoffs.
- `label` is `H/A/D`, or `X` where there is no label. BAR-ATH is labelled only on the public 5 minutes.
- A slot is a column pair, **not a person**. In BAR-ATH slots change person, so treat players as a set.
- `load_match(path)` returns numpy arrays: `ball (T,3)`, `home (T,S,2)`, `away (T,S,2)`, `label (T,)`.

## What's in each script

| file | does |
|---|---|
| [fetch_dfl.py](fetch_dfl.py) | download + verify + convert DFL |
| [convert_atd.py](convert_atd.py) | convert BAR-ATH. Prints only aggregates, never values |
| [schema.py](schema.py) | the shared format: paths, `write_match()`, `validate()`, `load_match()` |
| [labels.py](labels.py) | `controlled()`: merges short sandwiched DFL possessions into the surrounding one |
| [noise.py](noise.py) | aggregate tracking-noise profile of BAR-ATH (no frame-level values, no labels used for the player statistics): the `fragments` player corruption reads it |
| [datasets_visualization/](datasets_visualization/) | `shared.py`, `open_datasets.py`, `recipes.py`: the DFL readers used by the labelling app (`possession_exploration/app.py`) |

**Checks that passed:** the baseline rebuilt from `data/processed/BAR-ATH.parquet` reproduces the
published public-window scores exactly (0.597/0.038, 0.620/0.287, 0.621/0.326, 0.581/0.424). That
validates frame alignment and team mapping end to end.

## License and citation

DFL data: CC-BY 4.0. Cite it in the technical document:

> Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and
> event data in elite soccer. *Scientific Data*, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
