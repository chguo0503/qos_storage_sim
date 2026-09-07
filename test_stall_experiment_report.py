"""Audit the saved experimental evidence, including unfavorable cases."""
import unittest

from build_stall_experiment_report import CASES, STRATEGIES, collect


class ReportEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.selected = collect()

    def test_main_comparisons_have_common_time_seed_hash_and_all_cards_active(self):
        for case in CASES:
            rows = [self.selected[(*case, s)][0] for s in STRATEGIES]
            self.assertEqual(len({r["input_fingerprint"] for r in rows}), 1)
            self.assertEqual(len({r["submit_seed"] for r in rows}), 1)
            self.assertTrue(all(r["start_ms"] == 1000 and r["end_ms"] == 2000 for r in rows))
            self.assertTrue(all(r["all_npus_active_whole_window"] for r in rows))

    def test_unmatched_seed_exploration_is_preserved_not_used_as_matched_evidence(self):
        unmatched = [m for m, _ in self.records if not m["seed_matched"]]
        self.assertTrue(unmatched)
        selected_files = {m["file"] for m, _ in self.selected.values()}
        self.assertTrue(all(m["file"] not in selected_files for m in unmatched))

    def test_negative_outcome_is_retained(self):
        case = "long", 20260910
        once = self.selected[(*case, "layer_once")][0]
        deadline = self.selected[(*case, "dedicated_edf")][0]
        self.assertLess(deadline["mean_npu_utilization"], once["mean_npu_utilization"])


if __name__ == "__main__":
    unittest.main()
