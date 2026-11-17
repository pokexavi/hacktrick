# Synthetic samples — made-up values, real formats

**Nothing in this folder is match data.** Every number is produced by a random simulation in
`make_synthetic.py` (possessions, stoppages, 22 players, a ball with tracking gaps, fragmented track
slots, a panning camera). The files have exactly the format of the real bundle — same file names,
tag trees, key names, separators, field counts, `NaN`/`nan` tokens and number formatting — so code
written against them runs unchanged on `data/`.

The format was checked against the real files with a probe that printed only structure (tag paths
and counts, key names, field counts, decimals) and never values.

```
synthetic/
├─ tactical/     BAR-ATH_tactical_{atd.xml, atd.txt, homography.csv, smart_tagging.xml}   750 frames
└─ broadcast/    BAR-ATH_broadcast_{atd.xml, atd.txt, homography.csv}                      600 frames
```

Point any loader at it with `games.asset_paths(data_dir="exploration/synthetic")`.

| | regenerate |
|---|---|
| this 30-second sample (committed) | `python exploration/synthetic/make_synthetic.py` |
| a full-length bundle for testing pipelines (gitignored, ~400 MB) | `python exploration/synthetic/make_synthetic.py --full --out cache/synthetic_full --video` |

The full-length bundle uses the real frame counts and graded halves, adds
`BAR-ATH_synthetic_labels.csv` (the simulation's own H/A/D/X labels — synthetic, not ground truth)
and, with `--video`, a 1920×1080 mp4 of the first 19,000 frames with the frame number burned in.
Run any exploration notebook on it with `$env:MTAH_DATA_DIR = "cache/synthetic_full"`.

Differences you should not read anything into: the counts (750 frames, a handful of tags), the
team/slot split (23/23 here) and player behaviour. Speeds mirror the real quirk: the metadata says
`kph` but the values are m/s. (The real ball `s` is also unreliable; the synthetic one is not.)
