"""Audit diagram event labels against the deliberately small teaching examples."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

import build_staggering_story as drawing
from toy_staggering_model import get_story_data


class StoryFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_three_single_order_examples_change_only_the_named_input(self):
        with patch.object(drawing, "save"):
            fail = drawing.one_order("unused", 1, 4, "test")
            phase = drawing.one_order("unused", 3, 4, "test")
            budget = drawing.one_order("unused", 1, 8, "test")
        self.assertEqual([x["ready_s"] for x in (fail, phase, budget)], [7, 7, 7])
        self.assertEqual([x["stall_s"] for x in (fail, phase, budget)], [2, 0, 0])
        self.assertEqual([x["deadline_s"] for x in (fail, phase, budget)], [5, 7, 9])
        self.assertEqual(fail["queue_s"], budget["queue_s"])

    def test_failure_figure_marks_actual_event_ticks(self):
        with patch.object(drawing, "save") as saving:
            drawing.one_order("unused", 1, 4, "test")
        axis = saving.call_args.args[0].axes[0]
        self.assertEqual(list(axis.get_xticks()), [0, 1, 5, 6, 7, 10])
        texts = [x.get_text() for x in axis.texts]
        for expected in ("1 秒下单", "7 秒到齐", "订单排队 5 秒", "停工 2 秒"):
            self.assertIn(expected, texts)

    def test_clipped_compute_does_not_claim_shorter_recipe_duration(self):
        with patch.object(drawing, "save") as saving:
            drawing.chronological("unused", get_story_data()["success"],
                                  origin=0, end=11, headline="test", ticks=[0, 1, 11])
        texts = [x.get_text() for x in saving.call_args.args[0].axes[0].texts]
        self.assertNotIn("做菜2秒", texts)
        self.assertIn("做到图外", texts)
        self.assertIn("续", texts)

    def test_repeat_stalls_are_at_true_times_not_packed_after_compute(self):
        with patch.object(drawing, "save") as saving:
            drawing.chronological("unused", get_story_data()["failure"],
                                  origin=27, end=21, headline="test", ticks=[0, 1, 21])
        axis = saving.call_args.args[0].axes[0]
        stalled = [(p.get_x(), p.get_x()+p.get_width())
                   for p in axis.patches if p.get_hatch() == "///"]
        self.assertEqual(stalled, [(5, 7), (15, 17)])
        texts = [x.get_text() for x in axis.texts]
        for expected in ("0下单", "1下单", "7下单", "10下单", "11下单", "21下单"):
            self.assertIn(expected, texts)
        self.assertEqual(texts.count("空闲"), 2)


if __name__ == "__main__":
    unittest.main()
