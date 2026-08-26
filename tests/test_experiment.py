from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import experiment


class ExperimentHelperTests(unittest.TestCase):
    def test_final_qos_layout_has_256_paths_and_40_gbps(self):
        config = experiment.qos_config()
        self.assertEqual(len(config.path_cirs), 256)
        self.assertAlmostEqual(sum(config.path_cirs), 40.0)
        self.assertEqual(
            experiment.CATEGORY_CIR_GBPS,
            (20.0, 6.0, 8.0, 6.0),
        )

    def test_plot_writes_original_multi_series_chart(self):
        data = {
            "ssus": [8, 16],
            "series": [
                {"label": "Baseline", "style": "s-", "values": [40, 50]},
                {"label": "QoS", "style": "o-", "values": [60, 70]},
            ],
            "ylabel": "Average NPU Utilization (%)",
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            experiment.plot_results(data, output)
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
