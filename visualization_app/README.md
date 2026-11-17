# Visualization app

A Streamlit app to look at the final pipeline's outputs against the video, the 2D tracking and the
public ground truth. Four pages:

| page | reads | what for |
|---|---|---|
| **Data visualization** | the configured video, Smart Tagging and ATD; GT and baselines from `evaluation/`; prediction folders of `cache/possession_model/bar/` | the public 5-minute window: video, 2D pitch, GT and predictions lane by lane |
| **Dead-ball review** | every model folder in `cache/dead_alive/` (`DEAD_ALIVE_ROOT`) | the dead-ball model's stoppages: P(dead), whistles, Smart Tagging, clip player; opens on `ens_keeperhold3_all7_tagchains` |
| **Possession review** | every prediction folder in `cache/possession_model/bar/` (`POSSESSION_ROOT`) | official metrics on the public window (a pipeline check, never a way to choose a model), whole-match aggregates, timeline, disagreements, clip player; opens on `v1_frozen_annotated_1` |
| **Whole match** | `cache/dead_alive/` | the timeline of the model's stoppages over the 132 minutes and a review of possibly missed stoppages |

The model folders are written by `python -m dead_alive_detection.final` and
`python -m possession_model.predict` (see the root [README](../README.md)). `cache/` is gitignored,
so a fresh clone has no predictions until the pipeline has run.

From the repository root:

```bash
python -m pip install -r requirements-pipeline.txt
python -m streamlit run visualization_app/streamlit_app.py
```

Tests (synthetic data only, no match data is read):

```bash
python visualization_app/test_dead_ball_review.py
python visualization_app/test_missed_review.py
python visualization_app/test_possession_review.py
python visualization_app/test_whistle_review.py
```

## Configure automatic inputs

Edit `config.py`. All relative paths are relative to the repository root.
Use forward slashes or raw strings for Windows paths.

```python
TACTICAL_VIDEO_PATH = "data/YOUR_TACTICAL_VIDEO.mp4"
SMART_TAGGING_PATH = "data/YOUR_SMART_TAGGING.xml"
```

Replace these example filenames with your actual filenames, including subfolders.
Leave the video path empty to select a video manually. Leave the Smart Tagging path
empty to omit tagging events. Smart Tagging is loaded only from configuration.
No directory is scanned.
Configured paths are read only when you run the app; private inputs were not inspected
during development. The video excerpt is encoded in memory and cached by Streamlit;
first preparation can take time. The full video is not embedded in the page.

- Tactical MP4: the full original tactical match video, starting at original frame 0.
- Smart Tagging XML: full-match SportsCode intervals (`instance/start/end/code/label`),
  in seconds from the first tactical video frame. Tracking ATD XML is a different input
  and can be enabled separately in the optional ATD view.
- GT CSV: already configured as `evaluation/gt/BAR-ATH_possession_gt_public.csv`.
- Baseline CSVs: the four files in `evaluation/results/` are already configured.
  Add other predictions via the sidebar, or edit `BASELINE_PATHS`.

The default window is original frames [11185, 18685), at 25 FPS: 300 seconds,
starting at 447.4 seconds (07:27.4) in the full tactical video. Full-match annotations
are filtered to overlapping intervals, preserving their original frame coordinates.
Only public GT frames are compared; missing/unknown GT is X and is never graded.
The configured full video is automatically clipped to this window. With manual video
selection, the player seeks to the same window; set its original first frame to 0
for the full video or to the crop start for an already cropped clip.

## Fixed GT and full-match predictions

The GT is always the configured public CSV; the UI does not offer GT replacement.
Upload one or more full-match `frame,label` CSVs in the sidebar, or add named paths:

```python
PREDICTION_PATHS = {"My model": "outputs/predictions/my_model.csv"}
```

Keep original zero-based tactical video frames. The app selects the GT window;
there is no need to crop or renumber prediction rows. Missing frames display X.

## Optional ATD

Set both paths in `config.py`:

```python
ATD_METADATA_PATH = "data/tactical/YOUR_ATD_METADATA.xml"
ATD_RAW_PATH = "data/tactical/YOUR_ATD_RAW.txt"
```

Enable or disable **Show ATD pitch view** in the sidebar. There are no ATD upload controls.
Both paths default to empty; they are read only when the optional view is enabled.
The synchronized pitch shows home/away slots and the ball. Frame alignment is
`video_frame = frame_id - 1`; non-finite positions are treated as missing.
Slots are tracking fragments, not stable player identities. The ATD view uses
normalized pitch coordinates; projecting onto the video would require homography.

No homography overlay is included. Do not commit private inputs or derived media.


## Page and ball availability timeline

The possession timeline includes a `ball_data_missing` row, synchronized with the
same video cursor and click-to-seek controls. Orange marks missing ball coordinates,
green marks available coordinates, and grey means there is no ATD record.
The row loads from `ATD_RAW_PATH` independently of the optional pitch view; no ATD
XML is required. A visible ball requires finite x and y; speed availability does
not determine visibility. Consecutive original frame IDs are validated.
