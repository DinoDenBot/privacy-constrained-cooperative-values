import importlib.util
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("confirmatory", HERE / "run.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BundleTests(unittest.TestCase):
    def test_mobius_round_trip_identity(self):
        values = np.asarray([0, 1, 2, 4, 3, 7, 8, 15], dtype=float)
        dividends = MODULE.mobius(values, 3)
        reconstructed = dividends.copy()
        for bit in range(3):
            for mask in range(8):
                if mask & (1 << bit):
                    reconstructed[mask] += reconstructed[mask ^ (1 << bit)]
        np.testing.assert_allclose(reconstructed, values)

    def test_analysis_identity(self):
        values = np.asarray([0, 0.1, 0.2, 0.4, 0.3, 0.6, 0.8, 1.2])
        metrics, _, _ = MODULE.analyze(values, 3)
        self.assertLess(metrics["identity_residual"], 1e-12)

    def test_brier_range(self):
        params = MODULE.initialize(4, 3, 2, 1)
        x = np.zeros((5, 4))
        y = np.asarray([0, 1, 0, 1, 0])
        _, brier, _ = MODULE.utility(params, x, y)
        self.assertGreaterEqual(brier, 0)
        self.assertLessEqual(brier, 2)

    def test_weighted_update_norms(self):
        updates = [
            (np.asarray([3.0, 4.0]),),
            (np.asarray([0.0, 2.0]),),
        ]
        norm_sum, norm_max, aggregate_norm = MODULE.weighted_update_norms(
            updates, np.asarray([0.25, 0.75])
        )
        self.assertAlmostEqual(norm_sum, 2.75)
        self.assertAlmostEqual(norm_max, 1.5)
        self.assertAlmostEqual(aggregate_norm, np.linalg.norm([0.75, 2.5]))

    def test_scaling_slope_recovery(self):
        rows = []
        for scale in (0.0625, 0.125, 0.25, 0.5):
            rows.append(
                {
                    "utility": "negative_brier",
                    "update_scale": scale,
                    "alpha": 0.5,
                    "partition_seed": 7,
                    "checkpoint_epoch": 0,
                    "local_epochs": 20,
                    "absolute_midpoint_gap": 4 * scale**3,
                    "relative_midpoint_gap": 2 * scale**2,
                    "identity_residual": 0.0,
                }
            )
        slopes, inconclusive = MODULE.scaling_slopes(rows, "negative_brier", 0.5, 1e-14)
        self.assertEqual(len(slopes), 1)
        self.assertEqual(inconclusive, [])
        self.assertAlmostEqual(slopes[0]["absolute_log_slope"], 3.0)
        self.assertAlmostEqual(slopes[0]["relative_log_slope"], 2.0)

    def test_scaling_slope_reports_numerical_floor(self):
        rows = []
        for scale in (0.0625, 0.125, 0.25, 0.5):
            rows.append(
                {
                    "utility": "negative_brier",
                    "update_scale": scale,
                    "alpha": 0.5,
                    "partition_seed": 7,
                    "checkpoint_epoch": 0,
                    "local_epochs": 20,
                    "absolute_midpoint_gap": 0.0,
                    "relative_midpoint_gap": 0.0,
                    "identity_residual": 0.0,
                }
            )
        slopes, inconclusive = MODULE.scaling_slopes(rows, "negative_brier", 0.5, 1e-14)
        self.assertEqual(slopes, [])
        self.assertEqual(len(inconclusive), 1)


if __name__ == "__main__":
    unittest.main()
