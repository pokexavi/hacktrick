# Possession model

Says which team has the ball (H or A) on every live frame of BAR-ATH. The live frames come from the
dead-ball model (`dead_alive_detection/`). It is trained on the 7 DFL matches with our annotations
(`possession_exploration/`), corrupted to look like BAR.

**Final model:** `results/v1_frozen_annotated_1/`, trained with [configs/v1_frozen.toml](configs/v1_frozen.toml)
as it is. Every option is in that file; change one for a single run with `--set section.key=value`.

## Scripts

| script | what it does |
|---|---|
| `config.py` | loads the toml and the `--set` changes |
| `data.py` | the DFL matches (6 to develop, 1 for test) and their labels with the annotations |
| `augment.py` | makes DFL look like BAR: players lost and ball hidden the way BAR's tracking does |
| `features.py` | tracking to steps of 0.2 s: one token per player, one for the ball, one for context |
| `model.py` | the network (architecture explained at the top of the file) |
| `train.py` | training windows, loss, and prediction over a whole match |
| `decode.py` | the HSMM that turns per-step probabilities into whole possessions |
| `evaluate.py` | scores a match against its labels with the official metric |
| `run.py` | trains and scores leave-one-match-out; `--test` scores the test match once |
| `grid.py` | tries decoder settings on a finished run without retraining |
| `predict.py` | predicts BAR with a trained run |

## Train

From the `Hacktrick/` root, after `python external/fetch_dfl.py` once:

```powershell
python -m possession_model.run                                       # CPU (or GPU if there is one)
python -m possession_model.run --set runtime.device='cuda'           # force the GPU
python -m possession_model.run --set train.epochs=15 --set "train.seeds=[0, 1, 2]" --set name=<name>
```

Results go to `results/<run>/` (config and scores) and the weights to
`cache/possession_model/<run>/`. An interrupted run continues with
`--resume possession_model/results/<run>`.

## Predict BAR and make the submission

These read the match: run them yourself and never commit what they write. Without training, get
the weights first from the inference bundle (root README, section 7.4). The bundle also holds the
final dead/live mask; `python -m dead_alive_detection.infer` rebuilds the same mask from the
dead-ball weights.

```powershell
python -m zipfile -e assets/inference/Hacktrick-inference.zip .

python -m possession_model.predict --run possession_model/results/v1_frozen_annotated_1 `
    --match data/processed/BAR-ATH.parquet `
    --dead cache/dead_alive/ens_keeperhold3_all7_tagchains/frames.csv `
    --out cache/possession_model/bar/v1_frozen_annotated_1

python -m dead_alive_detection.submit --dead cache/dead_alive/ens_keeperhold3_all7_tagchains `
    --teams cache/possession_model/bar/v1_frozen_annotated_1/labels.csv --out outputs/submissions/Hacktrick.csv
```

To look at it: `python -m streamlit run visualization_app/streamlit_app.py`, page **Possession review**.
