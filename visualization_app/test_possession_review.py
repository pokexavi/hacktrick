"""Tests of the Possession review page on made-up labels and the synthetic ATD sample (no match data).

    python visualization_app/test_possession_review.py
"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))
import possession_review as lib  # noqa: E402

SYNTHETIC = ROOT / "exploration" / "synthetic" / "tactical"


def labels(text):
    return np.array(list(text), dtype="<U1")


class Logic(unittest.TestCase):
    def test_spells_split_on_dead_and_x(self):
        self.assertEqual(lib.spells(labels("HHDDHHAAXA")), [(0, 2, "H"), (4, 6, "H"), (6, 8, "A"), (9, 10, "A")])
        self.assertEqual(lib.spells(labels("HHAA"), 1, 3), [(1, 2, "H"), (2, 3, "A")])

    def test_window_metrics_perfect_and_flicker(self):
        gt = labels("X" * 5 + "H" * 100 + "A" * 100 + "D" * 50 + "H" * 100 + "X" * 5)
        perfect = lib.window_metrics(gt, gt, 0, len(gt))
        self.assertAlmostEqual(perfect["spellF1"], 1.0)
        self.assertEqual(perfect["possessions"], 3)
        flicker = gt.copy()
        flicker[50:53] = "A"
        self.assertLess(lib.window_metrics(flicker, gt, 0, len(gt))["spellF1"], 1.0)

    def test_compare_starts(self):
        gt = labels("H" * 50 + "A" * 50 + "D" * 20 + "H" * 60)
        pred = labels("H" * 55 + "A" * 45 + "D" * 20 + "H" * 30 + "A" * 30)
        s = lib.compare_starts(pred, gt, 0, len(gt))
        self.assertEqual(list(s.status), ["found", "found", "found", "spurious"])
        self.assertAlmostEqual(s.lag_s[1], 5 / 25)
        self.assertEqual(s.after[2], "D")

    def test_disagreements_and_match_metrics(self):
        gt = labels("H" * 50 + "A" * 50)
        pred = labels("H" * 70 + "A" * 30)
        wrong = lib.disagreements(pred, gt, 0, 100)
        self.assertEqual(wrong[["start", "stop", "gt", "pred"]].values.tolist(), [[50, 70, "A", "H"]])
        m = lib.match_metrics(labels("XHHHHDDAAX"))
        self.assertEqual(m["possessions"], 2)
        self.assertAlmostEqual(m["D"], 2 / 8)

    def test_runs_and_agreement(self):
        r = lib.runs(labels("XHHDDAX"))
        self.assertEqual(r.values.tolist(), [[0, 1, "X"], [1, 3, "H"], [3, 5, "D"], [5, 6, "A"], [6, 7, "X"]])
        a = lib.agreement(labels("HHHHDDAAX"), labels("HHAADDAAH"))
        self.assertAlmostEqual(a["same label"], 6 / 8)
        self.assertAlmostEqual(a["same live/dead"], 1.0)
        self.assertEqual(a["frames on the other team"], 2)

    def test_model_disagreements(self):
        a = labels("H" * 100 + "D" * 50 + "A" * 100)
        b = labels("H" * 60 + "A" * 40 + "D" * 50 + "A" * 100)          # 40 frames on the other team
        c = labels("H" * 100 + "D" * 30 + "A" * 120)                    # 20 frames live instead of dead
        d = lib.model_disagreements({"a": a, "b": b, "c": c}, min_frames=10)
        self.assertEqual(d[["start", "stop", "kind", "a", "b", "c"]].values.tolist(),
                         [[60, 100, "team", "H", "A", "H"], [130, 150, "live/dead", "D", "D", "A"]])
        self.assertEqual(len(lib.model_disagreements({"a": a, "b": b, "c": c}, min_frames=10, gap=40)), 1)
        self.assertTrue(lib.model_disagreements({"a": a}).empty)


class Page(unittest.TestCase):
    def test_page_runs_on_synthetic_data(self):
        from streamlit.testing.v1 import AppTest
        import config
        n = 750
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(0)
            gt = labels(("H" * 120 + "A" * 100 + "D" * 40 + "A" * 80 + "H" * 160 + "D" * 50 + "H" * 200)[:n])
            pd.DataFrame({"frame": np.arange(n), "label": gt}).to_csv(tmp / "gt.csv", index=False)
            for name, shift in (("model_a", 0), ("model_b", 12)):
                folder = tmp / "bar" / name
                folder.mkdir(parents=True)
                pred = np.roll(gt, shift)
                p_h = np.where(pred == "H", 0.9, 0.1) + rng.normal(0, 0.05, n)
                pd.DataFrame({"frame": np.arange(n), "p_h": p_h, "logit_h": 0.0, "label": pred}).to_csv(
                    folder / "frames.csv", index=False)
                (folder / "model.json").write_text('{"run": "%s", "models": ["a"], "dead": "x/ens/frames.csv", '
                                                   '"decoding": {"method": "hsmm"}}' % name)
            saved = {k: getattr(config, k) for k in ("POSSESSION_ROOT", "GT_PATH", "WINDOW_START_FRAME",
                                                      "WINDOW_END_FRAME", "TACTICAL_VIDEO_PATH", "ATD_RAW_PATH",
                                                      "ATD_METADATA_PATH", "POSSESSION_DEFAULT")}
            try:
                config.POSSESSION_ROOT = str(tmp / "bar")
                config.POSSESSION_DEFAULT = ["model_a", "model_b"]
                config.GT_PATH = str(tmp / "gt.csv")
                config.WINDOW_START_FRAME, config.WINDOW_END_FRAME = 0, n
                config.TACTICAL_VIDEO_PATH = ""
                config.ATD_RAW_PATH = str(SYNTHETIC / "BAR-ATH_tactical_atd.txt")
                config.ATD_METADATA_PATH = str(SYNTHETIC / "BAR-ATH_tactical_atd.xml")
                at = AppTest.from_file(str(APP_DIR / "pages" / "possession.py"), default_timeout=120)
                at.run()
                self.assertFalse(at.exception, at.exception)
                self.assertEqual(len(at.dataframe), 3)               # public window, whole match, agreement
                at.radio[0].set_value("Disagreements with GT (public window)").run()
                self.assertFalse(at.exception, at.exception)
                at.radio[0].set_value("Reference's possession starts").run()
                self.assertFalse(at.exception, at.exception)
                at.radio[0].set_value("Where the models disagree (whole match)").run()
                self.assertFalse(at.exception, at.exception)
                self.assertTrue(any("stretches where the 2 predictions" in c.value for c in at.caption))
                for show in ("live/dead", "both"):
                    at.radio[1].set_value(show).run()
                    self.assertFalse(at.exception, at.exception)
                at.radio[0].set_value("Time").run()
                self.assertFalse(at.exception, at.exception)
            finally:
                for k, v in saved.items():
                    setattr(config, k, v)


if __name__ == "__main__":
    unittest.main()
