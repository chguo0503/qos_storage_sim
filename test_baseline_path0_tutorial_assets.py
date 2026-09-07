"""Display transforms preserve service, logical endpoints, and prefetch causality."""

import copy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from baseline_path0_layout import DATA
from make_baseline_path0_tutorial_assets import (
    _bin_ssd_service,
    _configure_matplotlib,
    _make_npu_timeline_readable,
    _make_npu_0_10ms_zoom,
    _make_zoom_annotated,
    _npu2_read_example,
    _prefetch_gap_example,
    _relative_layer_timeline,
    _save_vector_and_png,
)
import matplotlib.pyplot as plt


def command(npu, start, end):
    return {"npu_id": npu, "ssd_start_time_ms": start, "ssd_end_time_ms": end}


class FigureOutputTests(unittest.TestCase):
    def test_outputs_create_and_use_the_figures_subdirectory(self):
        with TemporaryDirectory() as temporary:
            figures = Path(temporary) / "figures"
            fig, ax = plt.subplots()
            self.addCleanup(plt.close, fig)
            ax.plot([0, 1], [0, 1])
            with patch("make_baseline_path0_tutorial_assets.FIGURES", figures):
                _save_vector_and_png(fig, "layout_test")
            self.assertEqual({path.name for path in Path(temporary).iterdir()}, {"figures"})
            self.assertEqual({path.name for path in figures.iterdir()},
                             {"layout_test.pdf", "layout_test.png"})
            self.assertTrue(all(path.stat().st_size > 0 for path in figures.iterdir()))


class ServiceBinTests(unittest.TestCase):
    def test_partial_bins_and_idle_time_are_time_weighted(self):
        rows = [command(2, 0, 0.04), command(3, 0.04, 0.11), command(0, 0.17, 0.21)]
        edges, service, idle = _bin_ssd_service(rows, 0, 0, 0.2)
        self.assertEqual(edges, [0, 0.1, 0.2])
        self.assertAlmostEqual(service[2][0], 0.04)
        self.assertAlmostEqual(service[3][0], 0.06)
        self.assertAlmostEqual(service[3][1], 0.01)
        self.assertAlmostEqual(service[0][1], 0.03)
        self.assertAlmostEqual(idle[0], 0)
        self.assertAlmostEqual(idle[1], 0.06)
        self.assertAlmostEqual(sum(map(sum, service)) + sum(idle), 0.2)

    def test_carry_in_clipping_and_short_final_bin(self):
        rows = [command(1, 999.9, 1000.02), command(2, 1000.12, 1000.2)]
        edges, service, idle = _bin_ssd_service(rows, 1000, 0, 0.15)
        self.assertEqual(edges, [0, 0.1, 0.15])
        self.assertAlmostEqual(service[1][0], 0.02)
        self.assertAlmostEqual(service[2][1], 0.03)
        self.assertAlmostEqual(sum(map(sum, service)) + sum(idle), 0.15)

    def test_overlapping_physical_commands_are_rejected(self):
        with self.assertRaises(AssertionError):
            _bin_ssd_service([command(0, 0, 0.04), command(1, 0.02, 0.05)], 0, 0, 0.1)

    def test_invalid_window_or_bin_is_rejected(self):
        with self.assertRaises(ValueError):
            _bin_ssd_service([], 0, 0, 0.1, 0)
        with self.assertRaises(ValueError):
            _bin_ssd_service([], 0, 1, 0)


class LayerTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _configure_matplotlib()
        with (DATA / "result.json").open(encoding="utf-8") as handle:
            cls.result = json.load(handle)
        with (DATA / "request_layer_timeline.csv").open(encoding="utf-8", newline="") as handle:
            cls.raw_rows = list(csv.DictReader(handle))

    def test_relative_timeline_preserves_endpoints_and_carry_in(self):
        original = copy.deepcopy(self.raw_rows)
        rows = _relative_layer_timeline(self.result, self.raw_rows)
        origin = self.result["baseline"]["measurement_start_ms"]
        self.assertEqual(self.raw_rows, original)
        self.assertEqual(len(rows), len(self.raw_rows))
        self.assertTrue(any(r["release"] < 0 for r in rows))
        for row, raw in zip(rows, self.raw_rows):
            self.assertAlmostEqual(row["release"] + origin, float(raw["io_release_time_ms"]))
            self.assertAlmostEqual(row["ready"] + origin, float(raw["io_ready_time_ms"]))

    def test_gap_is_early_ready_and_compute_continues(self):
        rows = _relative_layer_timeline(self.result, self.raw_rows)
        example = _prefetch_gap_example(rows)
        self.assertAlmostEqual(example["handoff"], 26.853780008252215)
        self.assertAlmostEqual(example["gap_start"], 33.34308669656093)
        self.assertAlmostEqual(example["gap_end"], 39.24394481035279)
        self.assertAlmostEqual(example["gap_ms"], 5.900858113791855)
        previous = example["layers"][3]
        self.assertLessEqual(previous["compute_start"], example["gap_start"])
        self.assertAlmostEqual(previous["compute_end"], example["gap_end"])

    def test_zoom_shows_every_npu1_read_overlapping_its_window(self):
        rows = _relative_layer_timeline(self.result, self.raw_rows)
        example = _prefetch_gap_example(rows)
        lower, upper = example["layers"][3]["release"], 65.0
        visible = {(r["request_id"], r["layer"]) for r in rows if r["npu_id"] == 1
                   and min(upper, r["ready"]) > max(lower, r["release"])}
        self.assertEqual(visible, {(100020, layer) for layer in (3, 4, 5, 6)})

    def test_another_request_in_gap_must_not_be_called_no_outstanding_io(self):
        rows = _relative_layer_timeline(self.result, self.raw_rows)
        rows.append({"npu_id": 1, "request_id": 999999, "layer": 0,
                     "release": 34.0, "ready": 36.0})
        with self.assertRaisesRegex(AssertionError, "Another unfinished layer read"):
            _prefetch_gap_example(rows)

    def test_changed_prefetch_trigger_rejects_old_explanation(self):
        rows = _relative_layer_timeline(self.result, self.raw_rows)
        next(r for r in rows if r["request_id"] == 100020 and r["layer"] == 5)["release"] += 0.1
        with self.assertRaisesRegex(AssertionError, "prefetch triggers"):
            _prefetch_gap_example(rows)

    def test_impossible_ready_after_compute_start_is_rejected(self):
        rows = copy.deepcopy(self.raw_rows)
        rows[0]["io_ready_time_ms"] = float(rows[0]["compute_start_time_ms"]) + 1.0
        with self.assertRaisesRegex(AssertionError, "Invalid read/compute ordering"):
            _relative_layer_timeline(self.result, rows)

    def test_figure_explains_blue_blank_layer_identity_and_full_window(self):
        fig, _ = _make_npu_timeline_readable(self.result, self.raw_rows)
        self.addCleanup(plt.close, fig)
        overview, detail = fig.axes
        self.assertEqual(overview.get_xlim(), (0.0, 1000.0))
        self.assertAlmostEqual(detail.get_xlim()[0], 9.38738199390491)
        self.assertEqual(detail.get_xlim()[1], 65.0)
        labels = "\n".join(t.get_text() for t in fig.texts + detail.texts + detail.get_yticklabels())
        for phrase in ("不是 SSD 持续传输", "没有未完成的读取", "L3 读取", "L4 读取", "L5 读取",
                       "空白 5.901 ms", "NPU1 仍在算 L3", "不是无限层深预取"):
            self.assertIn(phrase, labels)

    def test_figure_panels_have_nonoverlapping_titles_and_labels(self):
        fig, _ = _make_npu_timeline_readable(self.result, self.raw_rows)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overview, detail = fig.axes
        self.assertGreater(overview.get_tightbbox(renderer).y0,
                           detail.get_tightbbox(renderer).y1 + 3.0)


class Figure4ReadabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _configure_matplotlib()
        with (DATA / "result.json").open(encoding="utf-8") as handle:
            cls.result = json.load(handle)
        with (DATA / "request_layer_timeline.csv").open(encoding="utf-8", newline="") as handle:
            cls.layers = list(csv.DictReader(handle))
        with (DATA / "physical_block_trace.csv").open(encoding="utf-8", newline="") as handle:
            cls.blocks = list(csv.DictReader(handle))

    def test_npu2_read_latency_is_not_release_period_or_service_time(self):
        example = _npu2_read_example(self.result, self.layers, self.blocks)
        for key, expected in {
            "release_ms": 7.561567879612994, "ready_ms": 20.42063639252956,
            "next_release_ms": 37.41813069606087, "latency_ms": 12.859068512916565,
            "period_ms": 29.856562816447877, "ready_slack_ms": 16.99749430353131,
            "service_span_ms": 12.683668732643127, "service_ms": 6.4159393310546875,
        }.items():
            self.assertAlmostEqual(example[key], expected)
        self.assertEqual(example["command_count"], 1529)

    def test_incomplete_layer_trace_cannot_supply_read_example(self):
        blocks = [r for r in self.blocks if not (r["request_id"] == "200010"
                  and r["layer"] == "2" and r["block_idx"] == "0")]
        self.assertEqual(len(blocks), len(self.blocks) - 1)
        with self.assertRaisesRegex(AssertionError, "full layer"):
            _npu2_read_example(self.result, self.layers, blocks)

    def test_actual_100ms_bins_conserve_each_npu_and_idle_time(self):
        origin = self.result["baseline"]["measurement_start_ms"]
        edges, service, idle = _bin_ssd_service(self.blocks, origin, 0, 100)
        for npu in range(4):
            expected = sum(max(0, min(100, float(r["ssd_end_time_ms"]) - origin)
                               - max(0, float(r["ssd_start_time_ms"]) - origin))
                           for r in self.blocks if int(r["npu_id"]) == npu)
            self.assertAlmostEqual(sum(service[npu]), expected)
        for i, idle_ms in enumerate(idle):
            self.assertAlmostEqual(sum(lane[i] for lane in service) + idle_ms, edges[i + 1] - edges[i])
        self.assertAlmostEqual(sum(map(sum, service)) + sum(idle), 100)

    def test_worked_examples_use_time_overlap_not_counts_or_busy_only_denominator(self):
        origin = self.result["baseline"]["measurement_start_ms"]
        edges, service, idle = _bin_ssd_service(self.blocks, origin, 0, 10)
        self.assertAlmostEqual(service[0][52] * 1000, 28.02908420562744)
        self.assertAlmostEqual(idle[52] * 1000, 71.97091579437256)
        self.assertAlmostEqual(service[2][80] * 1000, 50.35400390625)
        self.assertAlmostEqual(service[3][80] * 1000, 49.64599609375)
        self.assertAlmostEqual(idle[80], 0)
        # One NPU0 layer straddles 4.1 ms; do not assign the whole command/layer
        # to the bucket containing its start.
        self.assertAlmostEqual(service[0][40] * 1000, 3.763056324351854)
        self.assertAlmostEqual(service[0][41] * 1000, 24.266027881275586)

    def test_zoom_states_include_carry_in_and_clip_at_10ms(self):
        fig, report = _make_npu_0_10ms_zoom(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        self.assertAlmostEqual(report["npu_state_ms"][0]["compute_ms"], 4.65719593305967)
        for row in report["npu_state_ms"]:
            self.assertAlmostEqual(row["compute_ms"] + row["wait_ms"], 10)
        for row in report["npu_state_ms"][1:]:
            self.assertAlmostEqual(row["compute_ms"], 10)
        for ax in fig.axes:
            self.assertEqual(ax.get_xlim(), (0, 10))

    def test_annotated_plot_has_ready_marker_latency_and_distinct_period(self):
        fig, _ = _make_zoom_annotated(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        text = "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)
        for phrase in ("12.859 ms", "7.562", "20.421", "37.418", "HBM 到齐", "两次发起间隔"):
            self.assertIn(phrase, text)

    def test_annotated_figure_panels_do_not_overlap(self):
        fig, _ = _make_zoom_annotated(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for above, below in zip(fig.axes, fig.axes[1:]):
            self.assertGreater(above.get_tightbbox(renderer).y0,
                               below.get_tightbbox(renderer).y1 + 3.0)

    def test_0_10ms_figure_panels_do_not_overlap(self):
        fig, _ = _make_npu_0_10ms_zoom(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        self.assertGreater(fig.axes[0].get_tightbbox(renderer).y0,
                           fig.axes[1].get_tightbbox(renderer).y1 + 3.0)

    def test_release_labels_do_not_enter_the_neighbor_npu_lane(self):
        fig, _ = _make_npu_0_10ms_zoom(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        checked = 0
        for label in ax.texts:
            if not label.get_text().startswith("读L") or label.get_position()[1] > 3:
                continue
            lane = round(label.get_position()[1])
            bbox = label.get_window_extent(renderer)
            self.assertGreater(bbox.y0, ax.transData.transform((0, lane + 0.23))[1])
            self.assertLess(bbox.y1, ax.transData.transform((0, lane + 0.77))[1])
            checked += 1
        self.assertEqual(checked, 3)


if __name__ == "__main__":
    unittest.main()
