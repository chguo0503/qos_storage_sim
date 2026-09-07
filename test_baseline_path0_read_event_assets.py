"""Figure 4 shows SSU-side endpoints while retaining HBM consistency audits."""

import json
import unittest

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text

from baseline_path0_layout import DATA
from make_baseline_path0_tutorial_assets import _configure_matplotlib, _read_rows
from make_baseline_path0_read_event_assets import (
    layer_read_events,
    make_read_event_figure,
    selected_read_events,
)


# Independently checked against the original layer and physical-command CSVs.
# Tuple fields: NPU, request, layer, release, last SSD completion, HBM ready.
EXPECTED = {
    "0a": (0, 111, 6, 7.589133893471626, 10.128277994092059, 10.130559659894061),
    "0b": (0, 111, 7, 10.130559659894061, 26.878557063038897, 26.881343580182147),
    "1a": (1, 100020, 3, 9.38738199390491, 26.85052797883327, 26.853780008252215),
    "1b": (1, 100020, 4, 26.853780008252215, 33.33983466714199, 33.34308669656093),
    "2a": (2, 200010, 2, 7.561567879612994, 20.41727945893581, 20.42063639252956),
    "2b": (2, 200010, 3, 37.41813069606087, 50.27384227538369, 50.27719920897744),
    "3a": (3, 300010, 2, 7.557371712620807, 20.24104044526443, 20.24439737885818),
    "3b": (3, 300010, 3, 37.41393452906868, 50.097603261712306, 50.100960195306056),
}

EXPECTED_INTERVALS = {
    0: 2.541425766422435,
    1: 17.466398014347305,
    2: 29.856562816447877,
    3: 29.856562816447877,
}


class ReadEventDiagramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _configure_matplotlib()
        with (DATA / "result.json").open(encoding="utf-8") as handle:
            cls.result = json.load(handle)
        cls.layers = _read_rows(DATA / "request_layer_timeline.csv")
        cls.blocks = _read_rows(DATA / "physical_block_trace.csv")
        cls.events = layer_read_events(cls.result, cls.layers, cls.blocks)
        cls.selected = selected_read_events(cls.events)

    @staticmethod
    def _identity(event):
        return tuple(int(event[key]) for key in ("npu_id", "request_id", "layer"))

    @staticmethod
    def _text(fig):
        labels = [artist.get_text() for artist in fig.findobj(Text)]
        labels.extend(cell.get_text().get_text() for ax in fig.axes for table in ax.tables
                      for cell in table.get_celld().values())
        return "\n".join(labels)

    def test_selected_events_match_independently_checked_trace_times(self):
        self.assertEqual(set(self.selected), set(EXPECTED))
        for label, expected in EXPECTED.items():
            with self.subTest(event=label):
                event = self.selected[label]
                self.assertEqual(self._identity(event), expected[:3])
                for key, value in zip(("release_ms", "last_ssd_ms", "ready_ms"), expected[3:]):
                    self.assertAlmostEqual(event[key], value, places=8)
                self.assertTrue(event["complete_trace"])

    def test_event_window_preserves_carry_in_without_clamping_release(self):
        origin = float(self.result["baseline"]["measurement_start_ms"])
        expected = {
            self._identity(row)
            for row in self.layers
            if float(row["io_release_time_ms"]) - origin < 100.0
            and float(row["io_ready_time_ms"]) - origin > 0.0
        }
        self.assertEqual({self._identity(event) for event in self.events}, expected)
        self.assertEqual(len(self.events), len(expected), "one event per request/layer")
        self.assertTrue(any(event["release_ms"] < 0.0 for event in self.events))
        self.assertTrue(any(event["ready_ms"] > 100.0 for event in self.events))

    def test_selected_layer_volume_and_accumulated_ssd_service_match_profiles(self):
        volumes_mib = (1.1480712890625, 263.505859375, 262.796875, 262.796875)
        service_ms = (0.02802908420562744, 6.433248519897461,
                      6.4159393310546875, 6.4159393310546875)
        for label, event in self.selected.items():
            with self.subTest(event=label):
                npu = int(event["npu_id"])
                self.assertAlmostEqual(event["volume_gib"] * 1024, volumes_mib[npu], places=8)
                self.assertAlmostEqual(event["service_ms"], service_ms[npu], places=8)

    def test_ssd_completion_and_hbm_ready_remain_distinct(self):
        for label, event in self.selected.items():
            with self.subTest(event=label):
                self.assertLess(event["release_ms"], event["last_ssd_ms"])
                self.assertLess(event["last_ssd_ms"], event["ready_ms"])
                self.assertLess(event["ready_ms"] - event["last_ssd_ms"], 0.004)
                self.assertNotEqual(f"{event['last_ssd_ms']:.6f}", f"{event['ready_ms']:.6f}")
        event = self.selected["2a"]
        self.assertAlmostEqual(event["ready_ms"] - event["release_ms"], 12.859068512916565)
        self.assertAlmostEqual(event["last_ssd_ms"] - event["release_ms"], 12.855711579322815)

    def test_selected_pair_is_each_npus_next_release_not_ssd_completion(self):
        for npu, interval in EXPECTED_INTERVALS.items():
            with self.subTest(npu=npu):
                a, b = self.selected[f"{npu}a"], self.selected[f"{npu}b"]
                following = sorted(
                    (event for event in self.events
                     if int(event["npu_id"]) == npu and event["release_ms"] > a["release_ms"]),
                    key=lambda event: event["release_ms"],
                )
                self.assertEqual(self._identity(following[0]), self._identity(b))
                self.assertAlmostEqual(b["release_ms"] - a["release_ms"], interval, places=8)
        self.assertAlmostEqual(self.selected["2b"]["release_ms"] - self.selected["2a"]["last_ssd_ms"],
                               17.00085123712506, places=8)
        self.assertAlmostEqual(self.selected["3b"]["release_ms"] - self.selected["3a"]["last_ssd_ms"],
                               17.17289408380425, places=8)

    def test_missing_physical_command_in_selected_layer_is_rejected(self):
        # Cover both the small seven-command layer and a large 1529-command layer.
        for label in ("0a", "2a"):
            with self.subTest(event=label):
                identity = EXPECTED[label][:3]
                drop = next(i for i, row in enumerate(self.blocks) if self._identity(row) == identity)
                incomplete = self.blocks[:drop] + self.blocks[drop + 1:]
                with self.assertRaises(AssertionError):
                    selected_read_events(layer_read_events(self.result, self.layers, incomplete))

    def test_ssd_completion_later_than_hbm_ready_is_rejected(self):
        identity = EXPECTED["2a"][:3]
        indices = [i for i, row in enumerate(self.blocks) if self._identity(row) == identity]
        index = max(indices, key=lambda i: float(self.blocks[i]["ssd_end_time_ms"]))
        altered = list(self.blocks)
        altered[index] = dict(altered[index])
        origin = float(self.result["baseline"]["measurement_start_ms"])
        altered[index]["ssd_end_time_ms"] = origin + self.selected["2a"]["ready_ms"] + 0.001
        with self.assertRaises(AssertionError):
            selected_read_events(layer_read_events(self.result, self.layers, altered))

    def test_rendered_figure_has_eight_distinct_compute_and_read_rows(self):
        fig, report = make_read_event_figure(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        main = fig.axes[0]
        self.assertAlmostEqual(main.get_xlim()[0], 0.0)
        self.assertAlmostEqual(main.get_xlim()[1], 100.0)
        labels = [label.get_text() for label in main.get_yticklabels() if label.get_text()]
        self.assertEqual(len(labels), 8)
        for npu in range(4):
            own_labels = [label for label in labels if f"NPU{npu}" in label.replace(" ", "")]
            self.assertEqual(len(own_labels), 2, labels)
            self.assertTrue(any("计算" in label and "等待" in label for label in own_labels), own_labels)
            self.assertTrue(any("读取" in label for label in own_labels), own_labels)
            self.assertAlmostEqual(report["release_intervals_ms"][npu], EXPECTED_INTERVALS[npu], places=8)

    def test_rendered_four_column_table_matches_all_eight_ssu_events(self):
        fig, report = make_read_event_figure(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        # Matplotlib table text is not reliably returned by fig.findobj(Text).
        table_rows = []
        for ax in fig.axes:
            for table in ax.tables:
                rows = {}
                for (row, column), cell in table.get_celld().items():
                    rows.setdefault(row, {})[column] = cell.get_text().get_text()
                table_rows.extend([cells[column] for column in sorted(cells)] for cells in rows.values())
        self.assertTrue(table_rows, "figure needs an event-time table")
        self.assertTrue(all(len(row) == 4 for row in table_rows), table_rows)
        self.assertEqual(set(report["selected_events"]), set(EXPECTED))
        for label, expected in EXPECTED.items():
            with self.subTest(event=label):
                matching_rows = [row for row in table_rows if any(cell.strip().lower() == label for cell in row)]
                self.assertEqual(len(matching_rows), 1, (label, table_rows))
                normalized = "".join("".join(matching_rows[0]).split())
                self.assertIn(f"R{expected[1]}/L{expected[2]}", normalized)
                event = report["selected_events"][label]
                self.assertEqual(self._identity(event), expected[:3])
                for key, value in zip(("release_ms", "last_ssd_ms"), expected[3:5]):
                    self.assertAlmostEqual(event[key], value, places=8)
                    self.assertIn(f"{value:.6f}", normalized)

    def test_period_arrows_belong_to_their_own_read_lane_and_release_endpoints(self):
        fig, report = make_read_event_figure(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        ax = fig.axes[0]
        arrows = [text for text in ax.texts if isinstance(text, Annotation) and text.arrow_patch]
        self.assertEqual(len(arrows), 2)
        for npu, arrow in zip((2, 3), arrows):
            a, b = (report["selected_events"][f"{npu}{tag}"] for tag in "ab")
            self.assertAlmostEqual(arrow.xyann[0], a["release_ms"])
            self.assertAlmostEqual(arrow.xy[0], b["release_ms"])
            read_y = ax.get_yticks()[2*npu+1]
            compute_y = ax.get_yticks()[2*npu]
            self.assertGreater(arrow.xy[1], read_y)
            self.assertLess(arrow.xy[1], compute_y-0.23)
            self.assertAlmostEqual(arrow.xyann[1], arrow.xy[1])
        self.assertTrue(any("12.855712 ms" in text.get_text() for text in ax.texts))

    def test_ssu_figure_has_no_hbm_diamond_markers_or_labels(self):
        fig, _ = make_read_event_figure(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        for line in fig.findobj(Line2D):
            self.assertNotIn(line.get_marker(), ("D", "d"), "HBM diamond marker remains")
        compact_text = "".join(self._text(fig).split())
        for removed_label in ("◆", "◇", "HBM到齐", "符号下移"):
            self.assertNotIn(removed_label, compact_text)

    def test_npu2_latency_label_ends_at_ssd_completion_not_hbm_ready(self):
        fig, report = make_read_event_figure(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        event = report["selected_events"]["2a"]
        ssd_latency = event["last_ssd_ms"] - event["release_ms"]
        hbm_latency = event["ready_ms"] - event["release_ms"]
        self.assertAlmostEqual(ssd_latency, 12.855711579322815, places=8)
        self.assertGreater(hbm_latency, ssd_latency)
        label = next(text.get_text() for text in fig.axes[0].texts
                     if "2a" in text.get_text() and "○" in text.get_text()
                     and "■" in text.get_text())
        self.assertIn("○→■", "".join(label.split()))
        self.assertIn(f"{ssd_latency:.6f} ms", label)
        self.assertNotIn(f"{hbm_latency:.6f}", label)
        self.assertNotIn(f"{hbm_latency:.3f}", label)
        segments = [list(line.get_xdata()) for line in fig.axes[0].lines]
        self.assertTrue(any(len(x) == 2 and abs(x[0] - event["release_ms"]) < 1e-8
                            and abs(x[1] - event["last_ssd_ms"]) < 1e-8
                            for x in segments), "2a line must stop at last SSD completion")

    def test_event_table_and_timeline_do_not_overlap(self):
        fig, _ = make_read_event_figure(self.result, self.layers, self.blocks)
        self.addCleanup(plt.close, fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        self.assertGreater(fig.axes[0].get_tightbbox(renderer).y0,
                           fig.axes[1].get_tightbbox(renderer).y1 + 3)


if __name__ == "__main__":
    unittest.main()
