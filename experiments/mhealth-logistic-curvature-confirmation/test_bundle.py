import importlib.util
import json
import math
import sys
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/uci-har-logistic-curvature"))
import common  # noqa: E402


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "config.json").read_text())

    def test_roles_are_frozen_and_disjoint(self):
        self.assertEqual(len(self.config["profiles"]), 20)
        for roles in self.config["profiles"]:
            assigned = [roles["checkpoint_subject"], roles["test_subject"], *roles["players"]]
            self.assertEqual(sorted(assigned), list(range(1, 11)))
            self.assertEqual(len(set(assigned)), 10)

    def test_confirmatory_settings_are_fixed(self):
        self.assertEqual(self.config["epsilons"], [0.5])
        self.assertEqual(self.config["feature_radius"], 1.0)
        self.assertEqual(self.config["local_update_clip"], 0.2)
        self.assertEqual(self.config["scientific_gates"]["top_client_certification_rate_minimum"], 0.6)

    def test_all_inputs_are_hash_bound(self):
        hashes = json.loads((HERE / "input-hashes.json").read_text())
        self.assertEqual(len(hashes), 11)
        self.assertEqual(set(hashes), {"README.txt", *[f"mHealth_subject{i}.log" for i in range(1, 11)]})

    def test_certificate_covers_random_toy_game(self):
        rng = np.random.default_rng(319)
        n, dimension, classes = 4, 24, 12
        x = rng.normal(size=(120, dimension))
        x /= np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
        y = rng.integers(0, classes, len(x))
        checkpoint = rng.normal(size=(dimension, classes)) * 0.1
        caps = [0.02, 0.03, 0.04, 0.05]
        updates = []
        for cap in caps:
            update = rng.normal(size=(dimension, classes))
            updates.append(update * cap / np.linalg.norm(update))
        values = common.coalition_values(checkpoint, np.stack(updates), x, y, 0.5)
        radius = np.asarray(common.scalar_radii(caps, 0.5, 1 / math.sqrt(2)))
        error = np.abs(common.midpoint(values, n) - common.exact_shapley(values, n))
        self.assertTrue(np.all(error <= radius + 1e-12))


if __name__ == "__main__":
    unittest.main()
