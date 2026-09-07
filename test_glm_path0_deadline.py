"""只检验固定业务口径和计算结果，不测试输入防御。"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from glm_path0_deadline import (
    IO_SIZE_BYTES, KV_BLOCK_BYTES, estimate_path0_read,
)


class Path0ReadTests(unittest.TestCase):
    def test_fixed_units_and_example(self):
        self.assertEqual(KV_BLOCK_BYTES, 128 * 1408)
        self.assertEqual(KV_BLOCK_BYTES, 176 * 1024)
        self.assertEqual(IO_SIZE_BYTES, KV_BLOCK_BYTES)
        r = estimate_path0_read(10, 0.5, 100)
        self.assertEqual(r["required_io_count"], 10)
        self.assertEqual(r["queue_wait_ms"], 0.41961669921875)
        self.assertEqual(r["own_service_ms"], 0.041961669921875)
        self.assertEqual(r["estimated_read_ms"], 0.461578369140625)
        self.assertEqual(r["early_by_ms"], 0.038421630859375)
        self.assertEqual(r["late_by_ms"], 0)
        self.assertTrue(r["meets_deadline"])

    def test_deadline_changes_result_not_read_time(self):
        for deadline in (0.4, 0.461578369140625, 0.5):
            with self.subTest(deadline=deadline):
                r = estimate_path0_read(10, deadline, 100)
                self.assertEqual(r["estimated_read_ms"], 0.461578369140625)
                self.assertEqual(r["slack_ms"], deadline - 0.461578369140625)
                self.assertEqual(r["meets_deadline"], deadline >= 0.461578369140625)
                self.assertEqual(r["early_by_ms"], max(0, r["slack_ms"]))
                self.assertEqual(r["late_by_ms"], max(0, -r["slack_ms"]))

    def test_active_remainder_is_counted_once(self):
        r = estimate_path0_read(1, 0.1, 1, active_remaining_bytes=IO_SIZE_BYTES // 2)
        self.assertEqual(r["queue_wait_ms"], 1.5 * 0.0041961669921875)
        self.assertEqual(r["estimated_read_ms"], 2.5 * 0.0041961669921875)

    def test_zero_work_does_not_wait_for_old_queue(self):
        r = estimate_path0_read(0, 0, 100, active_remaining_bytes=100)
        self.assertEqual(r["estimated_read_ms"], 0)
        self.assertEqual(r["queue_wait_ms"], 0)
        self.assertEqual(r["required_io_count"], 0)
        self.assertTrue(r["meets_deadline"])

    def test_every_block_is_one_io_without_grouping_or_padding(self):
        for count in (1, 10, 128, 129):
            with self.subTest(count=count):
                r = estimate_path0_read(count, 10, 0)
                self.assertEqual(r["required_io_count"], count)
                self.assertNotIn("padded_kv_blocks", r)
                self.assertEqual(r["estimated_read_ms"], count * 0.0041961669921875)

    def test_empty_queue_and_bandwidth_scaling(self):
        fast = estimate_path0_read(10, 0.5, 0, bandwidth_gib_s=40)
        slow = estimate_path0_read(10, 0.5, 0, bandwidth_gib_s=20)
        self.assertEqual(fast["queue_wait_ms"], 0)
        self.assertEqual(fast["estimated_read_ms"], fast["own_service_ms"])
        self.assertEqual(slow["estimated_read_ms"], 2 * fast["estimated_read_ms"])

    def test_cli_matches_function_and_runs_as_a_copied_single_file(self):
        with tempfile.TemporaryDirectory(prefix="glm-path0-simple-") as tmp:
            copied = Path(tmp) / "glm_path0_deadline.py"
            shutil.copy2(Path(__file__).with_name("glm_path0_deadline.py"), copied)
            output = subprocess.check_output([
                sys.executable, str(copied), "--kv-block-count", "10",
                "--deadline-ms", "0.5", "--queued-io-count", "100",
            ], text=True, cwd=tmp)
            self.assertEqual(json.loads(output), estimate_path0_read(10, 0.5, 100))

    def test_cli_active_remainder_and_miss_are_normal_results(self):
        script = Path(__file__).with_name("glm_path0_deadline.py")
        output = subprocess.check_output([
            sys.executable, str(script), "--kv-block-count", "128",
            "--deadline-ms", "0.1", "--queued-io-count", "1",
            "--bandwidth-gib-s", "20", "--active-remaining-bytes", "1408",
        ], text=True)
        expected = estimate_path0_read(128, 0.1, 1, 20, 1408)
        self.assertEqual(json.loads(output), expected)
        self.assertFalse(expected["meets_deadline"])


if __name__ == "__main__":
    unittest.main()
