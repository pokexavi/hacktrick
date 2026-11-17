TACTICAL_VIDEO_PATH = "data/tactical/BAR-ATH_tactical.mp4"
BROADCAST_VIDEO_PATH = "data/broadcast/BAR-ATH_broadcast.mp4"  # its sound: the Whistles page
SMART_TAGGING_PATH = "data/tactical/BAR-ATH_tactical_smart_tagging.xml"
GT_PATH = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
BASELINE_PATHS = {
    name: f"evaluation/results/baseline_{name}.csv"
    for name in ("raw", "smooth25", "smooth50", "smooth125")
}
FPS = 25
# Original tactical video coordinates, end exclusive: exactly 5 minutes.
WINDOW_START_FRAME = 11185
WINDOW_END_FRAME = 18685


PREDICTION_PATHS = {}  # e.g. {"My model": "outputs/predictions/my_model.csv"}
ATD_METADATA_PATH = "data/tactical/BAR-ATH_tactical_atd.xml"
ATD_RAW_PATH = "data/tactical/BAR-ATH_tactical_atd.txt"
TACTICAL_HOMOGRAPHY_PATH = "data/tactical/BAR-ATH_tactical_homography.csv"
DEAD_ALIVE_ROOT = "cache/dead_alive"  # one subfolder per model: python -m dead_alive_detection.run / .predict
# Models selected when the Dead-ball review page opens (those that exist under DEAD_ALIVE_ROOT):
# the final dead-ball model (python -m dead_alive_detection.final)
DEAD_ALIVE_DEFAULT = ["ens_keeperhold3_all7_tagchains"]

POSSESSION_ROOT = "cache/possession_model/bar"  # one subfolder per prediction: python -m possession_model.predict
# Predictions selected when the Possession review page opens (those that exist under POSSESSION_ROOT)
POSSESSION_DEFAULT = ["v1_frozen_annotated_1"]  # the final possession model
