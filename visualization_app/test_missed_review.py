"""Missed-stoppage candidates on made-up arrays; never open match inputs."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import missed_review as mr

FPS = 25


def scene():
    """100 s, alive except a stoppage from 40 s to 50 s."""
    dead = np.zeros(100 * FPS, dtype=bool)
    dead[40 * FPS:50 * FPS] = True
    return dead, ~dead


class MissedReviewTests(unittest.TestCase):
    def test_runs_and_pieces(self):
        mask = np.array([0, 1, 1, 0, 1], dtype=bool)
        self.assertEqual(mr.runs_of(mask), [(1, 3), (4, 5)])
        self.assertEqual(mr.pieces(np.ones(250, dtype=bool), 4), [(0, 100), (100, 200), (200, 250)])

    def test_other_models_count_only_where_the_reference_is_alive(self):
        dead, alive = scene()
        voter = np.zeros_like(dead)
        voter[10 * FPS:13 * FPS] = True             # 3 s: a sign
        voter[20 * FPS:21 * FPS] = True             # 1 s: too short
        voter[42 * FPS:46 * FPS] = True             # the reference is Dead there
        signs = mr.model_signs(alive, np.stack([voter, np.zeros_like(voter)]))
        self.assertEqual([(s["start"], s["stop"]) for s in signs], [(10 * FPS, 13 * FPS)])
        self.assertEqual(signs[0]["strength"], 0.5)

    def test_p_dead_ignores_missing_values_and_short_bumps(self):
        _, alive = scene()
        p = np.full(alive.shape, np.nan)
        p[5 * FPS:6 * FPS] = 0.9
        p[60 * FPS:63 * FPS] = 0.5
        signs = mr.p_dead_signs(alive, p)
        self.assertEqual([s["start"] for s in signs], [60 * FPS])

    def test_whistles_on_dead_frames_are_not_signs(self):
        _, alive = scene()
        signs = mr.whistle_signs(alive, [10 * FPS, 45 * FPS, 10 ** 7], "preset")
        self.assertEqual([s["start"] for s in signs], [10 * FPS])

    def test_ball_gap_needs_five_seconds_alive(self):
        _, alive = scene()
        seen = np.ones(alive.shape)
        seen[10 * FPS:14 * FPS] = 0
        seen[20 * FPS:27 * FPS] = 0
        self.assertEqual([s["start"] for s in mr.ball_gap_signs(alive, seen)], [20 * FPS])

    def test_near_signs_join_and_more_kinds_rank_first(self):
        dead, alive = scene()
        signs = (mr.whistle_signs(alive, [70 * FPS], "preset")
                 + mr.p_dead_signs(alive, np.where(np.arange(alive.size) < 73 * FPS, 0, 0.5) * (np.arange(alive.size) < 76 * FPS))
                 + mr.whistle_signs(alive, [10 * FPS], "D20"))
        table = mr.candidates(signs, dead)
        self.assertEqual(len(table), 2)
        self.assertEqual(table["score"].tolist(), [2, 1])
        self.assertEqual(table["signs"][0], "p_dead, whistle")
        self.assertEqual(table["whistles"][1], "D20")
        self.assertEqual(table["to_stoppage_s"][1], 29.0)

    def test_a_tagged_event_names_what_it_overlaps_or_stands_alone(self):
        dead, alive = scene()
        signs = mr.whistle_signs(alive, [10 * FPS], "preset")
        tagged = pd.DataFrame({"start": [5 * FPS, 60 * FPS], "stop": [30 * FPS, 90 * FPS],
                               "kind": ["corner", "throw_in"]})
        table = mr.candidates(signs, dead, tagged).sort_values("start").reset_index(drop=True)
        self.assertEqual(table["tagged"].tolist(), ["corner", "throw_in"])
        self.assertEqual(table["score"].tolist(), [2, 1])
        self.assertEqual(table["signs"][1], "tagging")

    def test_long_alive_lists_the_longest_first(self):
        alive = np.zeros(400 * FPS, dtype=bool)
        alive[:100 * FPS] = True
        alive[110 * FPS:300 * FPS] = True
        self.assertEqual(mr.long_alive(alive)["seconds"].tolist(), [190.0, 100.0])

    def test_no_signs_give_an_empty_table(self):
        dead, _ = scene()
        self.assertEqual(list(mr.candidates([], dead).columns), mr.COLUMNS)
        self.assertEqual(len(mr.candidates([], dead)), 0)


if __name__ == "__main__":
    unittest.main()
