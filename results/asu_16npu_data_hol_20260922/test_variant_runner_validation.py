"""Small interpolation/schema/manifest tests; no simulation is executed."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("asu16_variant_test", HERE / "variant_runner.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def config():
    catalog = {"A": v.make_profile(32 * 1024, 4096, "A"), "B": v.make_profile(48 * 1024, 4096, "B")}
    sequences = []
    for card in range(16):
        key = f"adjust_{card}"
        catalog[key] = v.make_profile(32 * 1024 + card + 1, 3000 + card, "A")
        sequences.append([key, "A", "B"])
    return dict(name="variant_validation_only", id="variant_validation_only", num_npu=16,
        ssu=3, seed=7, horizon_ms=100, window_ms=[0, 50], mode="variant_explicit",
        profiles_catalog=catalog, per_npu_sequences=sequences)


class VariantValidation(unittest.TestCase):
    def test_all_grid_points_and_interpolation_edges(self):
        for length, miss in v.DATA:
            profile = v.make_profile(length * 1024, miss, "A")
            v.validate_profile(profile, "grid")
            self.assertFalse(profile["constructed_profile"])
            self.assertEqual(profile["compute_us"], v.DATA[length, miss][1])
        for tokens, miss, corners in ((32 * 1024, 3000, 2), (33 * 1024 + 17, 2048, 2),
                                       (199 * 1024 + 1, 4000, 4), (32 * 1024 + 1, 65, 4)):
            profile = v.make_profile(tokens, miss, "B")
            v.validate_profile(profile, "interpolated")
            self.assertEqual(len(profile["profile_construction"]["anchors"]), corners)
            self.assertAlmostEqual(sum(a["weight"] for a in profile["profile_construction"]["anchors"]), 1)

    def test_reject_extrapolation_and_unverifiable_values(self):
        for tokens, miss in ((32767, 64), (204801, 64), (32768, 63), (32768, 4097)):
            with self.assertRaises(ValueError):
                v.make_profile(tokens, miss, "A")
        mutations = {
            "scaled_C": lambda p: p.update(compute_us=p["compute_us"] * 1.1),
            "padding": lambda p: p.update(padding_gib_per_layer=0.1),
            "scaled_volume": lambda p: p.update(read_gib=p["read_gib"] * 1.1),
            "wrong_B": lambda p: p.update(B_gib_s=1),
            "wrong_sha": lambda p: p["profile_construction"].update(data_sha256="0" * 64),
            "missing_sha": lambda p: p["profile_construction"].pop("data_sha256"),
            "negative_weight": lambda p: p["profile_construction"]["anchors"][0].update(weight=-0.1),
            "wrong_weight": lambda p: p["profile_construction"]["anchors"][0].update(weight=0.9),
            "wrong_anchor": lambda p: p["profile_construction"]["anchors"][0].update(seq_len_k=80),
            "forged_anchor_C": lambda p: p["profile_construction"]["anchors"][0].update(compute_us=1),
            "wrong_constructed_label": lambda p: p.update(constructed_profile=False),
            "wrong_length_label": lambda p: p["profile_construction"].update(total_length_interpolated=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                profile = v.make_profile(33 * 1024, 3000, "A")
                mutate(profile)
                with self.assertRaises(ValueError):
                    v.validate_profile(profile, name)

    def test_sequence_rules_and_no_idle(self):
        good = config()
        normalized = v.validate_config(good)
        self.assertEqual(normalized["input_counts"], [2, 1])
        for case in ("idle", "unequal", "missing_B", "too_short"):
            bad = copy.deepcopy(good)
            if case == "idle": bad["per_npu_sequences"][0].append("idle")
            if case == "unequal": bad["per_npu_sequences"][0].append("A")
            if case == "missing_B": bad["per_npu_sequences"][0] = ["A", "A"]
            if case == "too_short": bad["horizon_ms"] = 1e8
            with self.subTest(case=case), self.assertRaises(ValueError):
                v.validate_config(bad)

    def test_build_manifest_roundtrip_and_identity(self):
        cfg = v.validate_config(config())
        requests, metadata, families = v.build(cfg)
        self.assertEqual(len(requests), 48)
        for card in range(16):
            self.assertEqual(families[card], ["A", "A", "B"])
            for position, key in enumerate(cfg["per_npu_sequences"][card]):
                row = metadata[card * 1000000 + position]
                self.assertEqual(row["original_request_id"], card * 1000000 + position)
                self.assertEqual(row["profile_key"], key)
                self.assertEqual(row["profile_group"], cfg["profiles_catalog"][key]["family"])
                self.assertEqual(row["arrival_ms"], 0)
                self.assertEqual(row["per_layer_us"], cfg["profiles_catalog"][key]["compute_us"])
        with tempfile.TemporaryDirectory(prefix="asu16_variant_validation_") as td:
            path = Path(td) / "manifest.json.gz"
            v.freeze_manifest(path, requests, cfg, families)
            loaded, _ = v.engine.load_manifest(path)
            self.assertEqual(v.engine.native.continuous_batch_input_fingerprint(loaded),
                             v.engine.native.continuous_batch_input_fingerprint(requests))
            manifest = v.engine.read_json(path)
            self.assertEqual(manifest["metadata"]["per_npu_profile_keys"], cfg["per_npu_sequences"])
            self.assertFalse(manifest["metadata"]["per_npu_equal_compute_work"])
            self.assertTrue(manifest["metadata"]["per_npu_equal_AB_counts"])
            self.assertEqual(len(manifest["metadata"]["per_npu_pure_compute_ms"]), 16)
            self.assertTrue(all(request.arrival_time_ms == 0 for request in loaded))


if __name__ == "__main__":
    unittest.main(verbosity=2)
