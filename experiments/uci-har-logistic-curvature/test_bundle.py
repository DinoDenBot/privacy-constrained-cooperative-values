import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


COMMON = load_module("bounded_logistic_common", HERE / "common.py")


class BoundedLogisticBundleTests(unittest.TestCase):
    def test_declared_game_count_and_primary_scale(self):
        config = json.loads((HERE / "config.json").read_text())
        self.assertEqual(len(config["role_seeds"]), 20)
        self.assertEqual(config["clients"], 8)
        self.assertIn(config["primary_epsilon"], config["epsilons"])

    def test_feature_projection_enforces_radius(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(40, 561)) * 100
        mean, scale = COMMON.fit_normalizer(features[:20])
        transformed = COMMON.bounded_features(features, mean, scale, 1.0)
        self.assertLessEqual(float(np.max(np.linalg.norm(transformed, axis=1))), 1 + 1e-12)
        self.assertEqual(transformed.shape, (40, 562))

    def test_categorical_third_central_moment_bound(self):
        rng = np.random.default_rng(20260910)
        bound = 1 / math.sqrt(2)
        for _ in range(1000):
            p = rng.dirichlet(np.ones(6))
            directions = [rng.normal(size=6) for _ in range(3)]
            directions = [value / np.linalg.norm(value) for value in directions]
            centered = [value - p @ value for value in directions]
            mixed = abs(float(p @ (centered[0] * centered[1] * centered[2])))
            self.assertLessEqual(mixed, bound + 1e-12)

    def test_public_cap_radius_covers_toy_logistic_game(self):
        rng = np.random.default_rng(41)
        n, dimension, classes = 4, 7, 3
        x = rng.normal(size=(80, dimension))
        x /= np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
        y = rng.integers(0, classes, size=len(x))
        checkpoint = rng.normal(size=(dimension, classes)) * 0.1
        caps = [0.03, 0.04, 0.05, 0.06]
        updates = []
        for cap in caps:
            value = rng.normal(size=(dimension, classes))
            value *= cap / np.linalg.norm(value)
            updates.append(value)
        values = COMMON.coalition_values(checkpoint, np.stack(updates), x, y, 0.8)
        truth = COMMON.exact_shapley(values, n)
        estimate = COMMON.midpoint(values, n)
        radius = np.asarray(COMMON.scalar_radii(caps, 0.8, 1 / math.sqrt(2)))
        self.assertTrue(np.all(np.abs(estimate - truth) <= radius + 1e-12))

    def test_constructor_source_excludes_private_inputs(self):
        source = (HERE / "construct.py").read_text()
        for forbidden in ("np.load", "--profiles", "--test-dir", "coalition_values(", "exact_shapley"):
            self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__":
    unittest.main()
