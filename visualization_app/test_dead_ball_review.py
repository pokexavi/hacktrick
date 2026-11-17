"""Dead-ball review helpers on hand-built tables; never open match inputs."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import dead_ball_review as review


class DeadBallReviewTests(unittest.TestCase):
    def test_smart_tagging_events(self):
        tags = [{"start": 30000, "end": 30875, "code": "SET PIECES", "tags": {"Type": "Corner Kick", "Half": "1st Half"}},
                {"start": 30000, "end": 30875, "code": "SET PIECES", "tags": {"Type": "Corner Kick", "Half": "1st Half"}},
                {"start": 20000, "end": 24000, "code": "GOALS", "tags": {"Half": "1st Half"}},
                {"start": 25000, "end": 26000, "code": "BUILD UP", "tags": {}}]
        events = review.smart_tagging_events(tags)
        self.assertEqual(list(events["kind"]), ["goal", "corner"])
        self.assertEqual(list(events.columns), ["kind", "team", "minute", "description", "start", "stop"])

    def test_summarise_and_coverage(self):
        stops = pd.DataFrame({"start": [100, 300, 400], "stop": [200, 350, 600],
                              "seconds": [4.0, 2.0, 8.0], "type": ["goal", "goal", "whistle"]})
        self.assertEqual(review.summarise(stops, 0, 50), "none")
        self.assertEqual(review.summarise(stops, 150, 160), "1 · 4 s goal")
        self.assertEqual(review.summarise(stops, 150, 450), "3 · 4+2+8 s goal/whistle")
        self.assertEqual(review.types_in(stops, 320, 410), ["goal", "whistle"])
        events = pd.DataFrame({"kind": ["goal", "goal"], "start": [150, 900], "stop": [160, 950]})
        table = review.coverage(events, {"m": stops})
        self.assertEqual(table.loc[0, "m"], "50% · 0.5")

    def test_without_stoppage_keeps_only_the_windows_with_none(self):
        stops = pd.DataFrame({"start": [100, 300], "stop": [200, 350]})
        events = pd.DataFrame({"kind": list("abcde"), "start": [0, 150, 199, 200, 350],
                               "stop": [100, 160, 210, 300, 400]})
        self.assertEqual(list(review.without_stoppage(events, stops)["kind"]), ["a", "d", "e"])
        everything = pd.DataFrame({"start": [0], "stop": [1000]})
        self.assertTrue(review.without_stoppage(events, everything).empty)
        self.assertEqual(list(review.without_stoppage(events, everything).columns), list(events.columns))

    def test_disagreement_kind(self):
        dead = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 0, 1, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0]], dtype=bool)
        self.assertEqual(review.disagreement_kind(dead, 2, 3), "gap inside a stoppage")
        self.assertEqual(review.disagreement_kind(dead, 0, 1), "boundary shifted")
        self.assertEqual(review.disagreement_kind(dead, 5, 6), "boundary shifted")
        self.assertEqual(review.disagreement_kind(dead, 6, 7), "stoppage seen by some models")

    def test_played_halves(self):
        play = review.in_play(199319)
        self.assertEqual(play.sum(), (90652 - 11185) + (190198 - 113742))
        self.assertFalse(play[110808:113742].any())
        self.assertEqual(review.half_bounds(50000), (11185, 90652))
        self.assertEqual(review.half_bounds(111000), (113742, 190198))   # pre-kickoff: nearest half
        stops = pd.DataFrame({"start": [11000, 50000, 110808, 189000], "stop": [11300, 50100, 113800, 190500],
                              "seconds": [12.0, 4.0, 119.68, 60.0]})
        out = review.clip_to_play(stops)
        self.assertEqual(out[["start", "stop"]].values.tolist(),
                         [[11185, 11300], [50000, 50100], [113742, 113800], [189000, 190198]])
        self.assertAlmostEqual(out["seconds"][2], 58 / 25)

    def test_describe_model(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder) / "m"
            folder.mkdir()
            (folder / "config.toml").write_text(
                '[augment.train]\nrecipe = "bar_motion"\n[types]\nclasses = ["goal", "other"]\nweight = 0.5\n'
                '[decoding.typed]\nenabled = false\n')
            (folder / "model.json").write_text('{"temperature": 0.25, "w_start": 0.5, "w_stop": 2.0}')
            row = review.describe_model(folder)
        self.assertEqual(row["trained on"], "bar_motion")
        self.assertIn("2 classes, weight 0.5", row["outputs"])
        self.assertEqual(row["decoder"], "HSMM (T 0.25, w_start 0.5, w_stop 2)")


if __name__ == "__main__":
    unittest.main()
