"""Deterministic checks for the client-side frozen hotspot forecast."""

from __future__ import annotations

import unittest

from forecast_hotspot_policy import forecast_frozen_ssu_hotspots


class FrozenHotspotForecastChecks(unittest.TestCase):
    def test_fleet_fallback_protects_every_ssu(self):
        forecast = forecast_frozen_ssu_hotspots((35.0, 28.0), hot_fraction=0.70)
        self.assertTrue(forecast.full_protection_fallback)
        self.assertEqual(forecast.materialized_ssu_mask, (True, True))

    def test_only_local_hotspots_are_selected_below_fleet_threshold(self):
        forecast = forecast_frozen_ssu_hotspots((30.0, 10.0), hot_fraction=0.70)
        self.assertFalse(forecast.full_protection_fallback)
        self.assertEqual(forecast.materialized_ssu_mask, (True, False))
        self.assertEqual(
            forecast.classification_by_ssu,
            ("hot_ssu", "cold_ssu_zero_cir"),
        )

    def test_authenticated_seed42_regime_known_answers(self):
        ssu6 = forecast_frozen_ssu_hotspots(
            (
                38.33101506501016,
                33.67009143544211,
                34.3556949518632,
                34.09822757030849,
                36.899475122839064,
                32.41378867893853,
            ),
            hot_fraction=0.70,
        )
        ssu10 = forecast_frozen_ssu_hotspots(
            (
                20.941230889707168,
                20.93190794371959,
                21.002981123460525,
                19.88139832353267,
                22.108302697739013,
                21.019399611646524,
                21.415275044797852,
                20.180267657953944,
                21.810042748275222,
                20.477486783569162,
            ),
            hot_fraction=0.70,
        )
        ssu18 = forecast_frozen_ssu_hotspots(
            (
                11.56070796607409,
                11.846636461567227,
                11.502217686664396,
                11.882294878619078,
                11.649408908544155,
                12.267950602608384,
                11.386979173449642,
                10.414643320739435,
                11.687120351069488,
                10.855056335724417,
                11.609063454689016,
                12.323410684298144,
                11.980958928036502,
                12.068668381061752,
                12.288346824832297,
                11.866796240641708,
                11.16961104954854,
                11.408421576233291,
            ),
            hot_fraction=0.70,
        )
        self.assertAlmostEqual(ssu6.fleet_load_fraction, 0.8740345534, places=9)
        self.assertEqual(ssu6.materialized_ssu_mask, (True,) * 6)
        self.assertEqual(ssu10.materialized_ssu_mask, (False,) * 10)
        self.assertEqual(ssu18.materialized_ssu_mask, (False,) * 18)

    def test_fingerprint_changes_with_threshold_or_demand(self):
        a = forecast_frozen_ssu_hotspots((20.0, 20.0), hot_fraction=0.70)
        b = forecast_frozen_ssu_hotspots((20.0, 20.0), hot_fraction=0.71)
        c = forecast_frozen_ssu_hotspots((20.0, 20.1), hot_fraction=0.70)
        self.assertNotEqual(a.input_fingerprint, b.input_fingerprint)
        self.assertNotEqual(a.input_fingerprint, c.input_fingerprint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
