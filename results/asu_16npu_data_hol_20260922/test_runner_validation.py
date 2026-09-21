"""Small source/order regressions. This file never runs the simulator."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("asu16_validation_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def direct(length, miss):
    compute = runner.DATA[length, miss][1]
    volume = (length * 1024 - miss) * 1408 / 2**30
    return dict(total_tokens=length * 1024, total_length_k=length, nql=miss,
        ssd_prefix_tokens=length * 1024 - miss, compute_us=compute, read_gib=volume,
        B_gib_s=volume / (compute / 1e6), constructed_profile=False,
        profile_construction=dict(method="exact_data_row", source="data",
            data_sha256=runner.DATA_SHA256, source_key=[length, miss],
            original_row=list(runner.DATA[length, miss]), compute_scale=1, extrapolated=False))


def config(a=None, b=None):
    return dict(name="validation_only", id="validation_only", num_npu=16, ssu=3,
        seed=7, horizon_ms=100, window_ms=[0, 50], mode="random", input_counts=[1, 2],
        profile_A=a or direct(32, 4096), profile_B=b or direct(48, 4096))


class ProfileValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.screen = json.loads((HERE / "screen_interpolated.json").read_text())
        cls.profile = cls.screen["shortlist"][0]["profile_A"]

    def test_every_screen_profile_and_original_row(self):
        for row in self.screen["shortlist"]:
            with self.subTest(candidate=row["name"]):
                runner.validate_config(config(row["profile_A"], row["profile_B"]))
        runner.validate_config(config())

    def test_reject_unverifiable_interpolation(self):
        mutations = {
            "negative_weight": lambda p: p["profile_construction"]["anchors"][0].update(weight=-0.1),
            "weights_not_sum_one": lambda p: p["profile_construction"]["anchors"][1].update(weight=0.5),
            "wrong_sha": lambda p: p["profile_construction"].update(data_sha256="0" * 64),
            "missing_sha": lambda p: p["profile_construction"].pop("data_sha256"),
            "fake_anchor_key": lambda p: p["profile_construction"]["anchors"][0].update(nql=2047),
            "nonadjacent_anchor": lambda p: p["profile_construction"]["anchors"][0].update(nql=1024),
            "different_length": lambda p: p["profile_construction"]["anchors"][0].update(seq_len_k=192),
            "duplicate_anchor": lambda p: p["profile_construction"]["anchors"].__setitem__(1, copy.deepcopy(p["profile_construction"]["anchors"][0])),
            "fake_anchor_compute": lambda p: p["profile_construction"]["anchors"][0].update(compute_us=1.),
            "fake_embedded_row": lambda p: p["profile_construction"]["anchors"][0]["original_data_row"].__setitem__(0, 1.),
            "extrapolated": lambda p: p["profile_construction"].update(extrapolated=True),
            "length_interpolated": lambda p: p["profile_construction"].update(total_length_interpolated=True),
            "compute_scale": lambda p: p["profile_construction"].update(compute_scale=1.01),
            "arbitrary_compute": lambda p: p.update(compute_us=p["compute_us"] + 1),
            "fake_volume": lambda p: p.update(read_gib=p["read_gib"] + 0.001),
            "outside_grid": lambda p: p.update(nql=4097),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                profile = copy.deepcopy(self.profile)
                mutate(profile)
                with self.assertRaises(ValueError):
                    runner.validate_profile_source(profile, "A")

    def test_reject_changed_direct_row(self):
        for field, value in (("compute_us", 1.), ("constructed_profile", True), ("read_gib", 1.)):
            with self.subTest(field=field):
                profile = direct(32, 4096)
                profile[field] = value
                with self.assertRaises(ValueError):
                    runner.validate_profile_source(profile, "B")

    def test_random_manifest_remains_byte_exact(self):
        old = runner._BASE_VALIDATE(config())
        new = runner.validate_config(config())
        self.assertEqual(old, new)
        original = runner._BASE_BUILD(old)
        current = runner.build(new)
        self.assertEqual(original, current)
        with tempfile.TemporaryDirectory(prefix="asu16_validation_") as td:
            old_path, new_path = Path(td) / "old.json.gz", Path(td) / "new.json.gz"
            runner._BASE_FREEZE(old_path, original[0], old, original[2])
            runner.freeze_manifest(new_path, current[0], new, current[2])
            self.assertEqual(old_path.read_bytes(), new_path.read_bytes())

    def test_explicit_and_cyclic_order(self):
        cfg = config()
        cfg.update(mode="cyclic", cycle="ABB", cycle_offsets=[i % 3 for i in range(16)])
        validated = runner.validate_config(cfg)
        decks = runner.controlled_decks(validated)
        self.assertEqual(decks[0], list("ABBABB"))
        self.assertEqual(decks[1], list("BBABBA"))
        explicit = runner.validate_config(dict(config(), mode="explicit", decks=decks))
        self.assertEqual(runner.build(validated), runner.build(explicit))
        with self.assertRaises(ValueError):
            runner.validate_config(dict(config(), mode="explicit", decks=["AAA"] * 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
