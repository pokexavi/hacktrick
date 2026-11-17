"""Checks for whistle_review.py, on hand-built whistles and stoppages only."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import whistle_review as wr  # noqa: E402

STOPS = pd.DataFrame({"start": [20000, 40000], "stop": [20500, 41000], "seconds": [20.0, 40.0]})


class WhistleReviewTests(unittest.TestCase):
    def test_categories_against_a_model(self):
        frames = np.array([20000 - 25, 20500 + 50, 20250, 30000, 5000])
        cat, to_start = wr.categorise(frames, STOPS)
        self.assertEqual(list(cat), ["near start", "near stop", "inside dead", "inside alive", "not graded"])
        self.assertAlmostEqual(to_start[0], -1.0)

    def test_summary_counts_starts_with_a_whistle(self):
        frames = np.array([20010, 30000])
        cat, _ = wr.categorise(frames, STOPS)
        s = wr.summary(frames, cat, STOPS)
        self.assertEqual((s["whistles"], s["starts_hit"], s["starts"]), (2, 1, 2))
        self.assertAlmostEqual(s["near_start"], 0.5)
        self.assertGreater(s["enrichment"], 1)

    def test_starts_without_a_whistle(self):
        left = wr.starts_without_whistle(STOPS, np.array([20010]))
        self.assertEqual(list(left["start"]), [40000])

    def test_curves_pool_the_frames(self):
        t = np.arange(100) / 100
        feat = {"t": t, "snr": np.arange(100.0), "prom": np.zeros(100), "conc": np.zeros(100),
                "peak": np.full(100, 3800.0)}
        c = wr.curves(feat, 0.0, 0.5, step=4)
        self.assertEqual(len(c["t"]), 12)
        self.assertEqual(c["snr"][0], 3.0)

    def test_mmss(self):
        self.assertEqual(wr.mmss(25 * 61), "1:01")

    def test_match_rows_joins_within_tolerance(self):
        table = pd.DataFrame({"frame": [1000, 5000]})
        self.assertEqual(list(wr.match_rows([1005, 3000, 4990], table)), [0, -1, 1])

    def test_spectrogram_columns_are_pooled(self):
        import tempfile
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tone.wav")
            t = np.arange(16000 * 4) / 16000
            wavfile.write(path, 16000, (np.sin(2 * np.pi * 3800 * t) * 8000).astype(np.int16))
            png, _, _ = wr.spectrogram_png(path, 0.0, 4.0, max_cols=100)
            from PIL import Image
            import base64, io
            self.assertLessEqual(Image.open(io.BytesIO(base64.b64decode(png))).size[0], 100)


if __name__ == "__main__":
    unittest.main()
