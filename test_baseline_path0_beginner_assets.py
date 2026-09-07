"""Teaching diagrams must derive their events and keep measurement scopes distinct."""

import json
import unittest

import matplotlib.pyplot as plt
from matplotlib.text import Text

from baseline_path0_layout import DATA
from make_baseline_path0_tutorial_assets import _configure_matplotlib, _read_rows
from make_baseline_path0_beginner_assets import (
    cycle_report, make_cycle_ledger, make_npu0_causal_story, make_npu1_causal_story,
    make_estimator_examples, make_prefetch_budget,
)


class BeginnerDiagramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _configure_matplotlib()
        with (DATA / "result.json").open(encoding="utf-8") as handle:
            cls.result = json.load(handle)
        with (DATA / "tutorial_numeric_audit.json").open(encoding="utf-8") as handle:
            cls.audit = json.load(handle)
        cls.rows = _read_rows(DATA / "request_layer_timeline.csv")

    def _text(self, fig):
        labels = [t.get_text() for t in fig.findobj(Text)]
        # Table cell labels are not reliably included by Artist.findobj(Text).
        labels.extend(cell.get_text().get_text() for ax in fig.axes for table in ax.tables
                      for cell in table.get_celld().values())
        return "\n".join(labels)

    def test_cycle_origin_is_real_release_pair_and_counts_are_equivalent_work(self):
        r = cycle_report(self.result, self.rows)
        self.assertAlmostEqual(r["start_ms"], 7.557371712620807)
        self.assertAlmostEqual(r["end_ms"], 37.41393452906868)
        self.assertAlmostEqual(r["period_ms"], 29.856562816447877)
        self.assertEqual([lane["equivalent_layers"] for lane in r["lanes"]], [11, 2, 1, 1])
        for lane in r["lanes"]:
            self.assertAlmostEqual(lane["compute_ms"] + lane["wait_ms"], r["period_ms"])
        self.assertAlmostEqual(r["lanes"][0]["wait_ms"], 23.452918408488586)
        self.assertAlmostEqual(r["lanes"][1]["wait_ms"], 5.076233212246734)

    def test_cycle_figure_explains_nonchronological_totals_and_not_request_counts(self):
        fig, _ = make_cycle_ledger(self.result, self.rows)
        self.addCleanup(plt.close, fig)
        text = self._text(fig)
        for phrase in ("累计", "不表示真实先后顺序", "不是完整请求个数", "7.557372", "37.413935", "29.856563"):
            self.assertIn(phrase, text)

    def test_npu0_event_letters_and_wait_lengths_match_trace(self):
        fig, r = make_npu0_causal_story(self.result, self.rows)
        self.addCleanup(plt.close, fig)
        expected = {"a": 7.589133893471626, "b": 8.171283385104289,
                    "c": 10.130559659894061, "d": 10.712709151526724, "e": 26.881343580182147}
        text = self._text(fig)
        for key, time in expected.items():
            self.assertAlmostEqual(r[key], time)
            self.assertIn(f"{time:.6f}", text)
        self.assertAlmostEqual(r["wait_l6_ms"], 1.959276274789772)
        self.assertAlmostEqual(r["wait_l7_ms"], 16.168634428655423)

    def test_npu1_uses_window_origin_and_tail_is_explicitly_microseconds(self):
        fig = make_npu1_causal_story(self.result, self.audit)
        self.addCleanup(plt.close, fig)
        text = self._text(fig)
        for time in (9.38738199390491, 20.41727945893581, 21.77754679600548,
                     26.85052797883327, 26.853780008252215):
            self.assertIn(f"{time:.6f}", text)
        self.assertIn("3.252029 µs", text)
        self.assertIn("不是从 a 重新计时", text)
        main, tail, table = fig.axes
        self.assertGreater(main.get_xlim()[0], 9)
        self.assertLess(tail.get_xlim()[1], 4)

    def test_npu1_event_diagram_panels_and_table_are_separate(self):
        fig = make_npu1_causal_story(self.result, self.audit)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for above, below in zip(fig.axes, fig.axes[1:]):
            self.assertGreater(above.get_tightbbox(renderer).y0, below.get_tightbbox(renderer).y1 + 3)

    def test_npu0_plot_and_event_table_are_separate(self):
        fig, _ = make_npu0_causal_story(self.result, self.rows)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        self.assertGreater(fig.axes[0].get_tightbbox(renderer).y0, fig.axes[1].get_tightbbox(renderer).y1 + 3)

    def test_hypothetical_examples_are_labeled_and_match_input_arithmetic(self):
        fig = make_estimator_examples(self.result)
        self.addCleanup(plt.close, fig)
        text = self._text(fig)
        for phrase in ("是预测", "不是已采集", "1.100", "6.278029", "5.695880", "0.028029"):
            self.assertIn(phrase, text)

    def test_prefetch_counts_match_compute_and_hbm_budget(self):
        fig = make_prefetch_budget(self.result)
        self.addCleanup(plt.close, fig)
        text = self._text(fig)
        p = self.result["input"]["profiles"][0]
        for count in (2, 29, 30):
            self.assertIn(f"{count * p['per_layer_compute_us'] / 1000:.6f}", text)
            self.assertIn(f"{count * p['per_layer_kv_gib'] * 1024:.3f}", text)
        self.assertIn("未实现或验证", text)

    def test_nearby_event_tick_labels_do_not_overlap(self):
        for fig in (make_npu1_causal_story(self.result, self.audit), make_estimator_examples(self.result)):
            self.addCleanup(plt.close, fig)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                ticks = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
                boxes = [t.get_window_extent(renderer) for t in ticks]
                for a, b in zip(boxes, boxes[1:]):
                    self.assertFalse(a.overlaps(b), [t.get_text() for t in ticks])


if __name__ == "__main__":
    unittest.main()
