# v1_frozen_annotated_1

Leave-one-match-out, mean over held-out matches. Headline condition: `bar_motion`. Reference: controlled DFL labels (min_frames = 25) with the manual logs of `possession_exploration/labels`.

| dataset | condition | spell_f1_mean | macro_f1 | team_acc | trans_f1_05s | trans_f1_1s | lag_1s | n_gt | n_pred |
|---|---|---|---|---|---|---|---|---|---|
| dfl | bar_motion | 0.6883 | 0.9373 | 0.9075 | 0.6909 | 0.7498 | 0.1244 | 399.3333 | 392.8333 |
| dfl | bar_motion_crowd | 0.6885 | 0.9383 | 0.9089 | 0.6911 | 0.7491 | 0.1189 | 399.3333 | 387.8333 |
| dfl | clean | 0.7025 | 0.9414 | 0.9134 | 0.7065 | 0.7588 | 0.1148 | 399.3333 | 395.5 |
