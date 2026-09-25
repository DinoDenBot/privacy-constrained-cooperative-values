#!/usr/bin/env python3
"""Exact fixed-update FL games for a deterministic nonlinear Fashion-MNIST model."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import kendalltau, spearmanr


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic == 2049:
            return np.frombuffer(handle.read(), dtype=np.uint8).copy()
        if magic != 2051:
            raise ValueError(f"unsupported IDX magic {magic} in {path}")
        rows, columns = struct.unpack(">II", handle.read(8))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, rows, columns).copy()


def stratified_subset(y: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = []
    per_class = count // len(np.unique(y))
    for label in np.unique(y):
        groups.append(rng.choice(np.flatnonzero(y == label), per_class, replace=False))
    return np.sort(np.concatenate(groups))


def prepare(data: Path, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = load_idx(data / "train-images-idx3-ubyte.gz").astype(np.float64) / 255.0
    train_y = load_idx(data / "train-labels-idx1-ubyte.gz").astype(np.int64)
    test_x = load_idx(data / "t10k-images-idx3-ubyte.gz").astype(np.float64) / 255.0
    test_y = load_idx(data / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    train_index = stratified_subset(train_y, int(config["train_examples"]), 1103)
    test_index = stratified_subset(test_y, int(config["evaluation_examples"]), 2203)
    train_x, train_y = train_x[train_index], train_y[train_index]
    test_x, test_y = test_x[test_index], test_y[test_index]
    factor = int(config["downsample_factor"])

    def pool(x: np.ndarray) -> np.ndarray:
        side = x.shape[1]
        return x.reshape(len(x), side // factor, factor, side // factor, factor).mean((2, 4)).reshape(len(x), -1)

    train_x, test_x = pool(train_x), pool(test_x)
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale == 0] = 1
    return (train_x - mean) / scale, train_y, (test_x - mean) / scale, test_y


def initialize(inputs: int, hidden: int, classes: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0, 1 / math.sqrt(inputs), (inputs, hidden)),
        np.zeros(hidden),
        rng.normal(0, 1 / math.sqrt(hidden), (hidden, classes)),
        np.zeros(classes),
    )


def probabilities(params: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2 = params
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2
    logits -= logits.max(1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


def train(
    params: tuple[np.ndarray, ...], x: np.ndarray, y: np.ndarray, epochs: int, rate: float, l2: float
) -> tuple[np.ndarray, ...]:
    result = tuple(value.copy() for value in params)
    targets = np.eye(result[-1].size)[y]
    for _ in range(epochs):
        w1, b1, w2, b2 = result
        hidden = np.tanh(x @ w1 + b1)
        logits = hidden @ w2 + b2
        logits -= logits.max(1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(1, keepdims=True)
        dlogits = (probs - targets) / len(x)
        dw2 = hidden.T @ dlogits + l2 * w2
        db2 = dlogits.sum(0)
        dhidden = (dlogits @ w2.T) * (1 - hidden**2)
        dw1 = x.T @ dhidden + l2 * w1
        db1 = dhidden.sum(0)
        result = (w1 - rate * dw1, b1 - rate * db1, w2 - rate * dw2, b2 - rate * db2)
    return result


def subtract(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(a - b for a, b in zip(left, right))


def scale_update(update: tuple[np.ndarray, ...], factor: float) -> tuple[np.ndarray, ...]:
    return tuple(factor * value for value in update)


def update_norm(update: tuple[np.ndarray, ...]) -> float:
    return float(math.sqrt(sum(float(np.square(value).sum()) for value in update)))


def weighted_update_norms(
    updates: list[tuple[np.ndarray, ...]], weights: np.ndarray
) -> tuple[float, float, float]:
    individual = [abs(float(weights[i])) * update_norm(update) for i, update in enumerate(updates)]
    aggregate = tuple(
        sum(
            (weights[i] * updates[i][part] for i in range(len(updates))),
            np.zeros_like(updates[0][part]),
        )
        for part in range(len(updates[0]))
    )
    return float(sum(individual)), float(max(individual)), update_norm(aggregate)


def add_weighted(
    checkpoint: tuple[np.ndarray, ...], updates: list[tuple[np.ndarray, ...]], weights: np.ndarray, mask: int
) -> tuple[np.ndarray, ...]:
    return tuple(
        base + sum((weights[i] * updates[i][part] for i in range(len(updates)) if mask & (1 << i)), np.zeros_like(base))
        for part, base in enumerate(checkpoint)
    )


def utility(params: tuple[np.ndarray, ...], x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    probs = probabilities(params, x)
    targets = np.eye(probs.shape[1])[y]
    cross_entropy = -float(np.log(probs[np.arange(len(y)), y] + 1e-15).mean())
    brier = float(np.square(probs - targets).sum(1).mean())
    accuracy = float((probs.argmax(1) == y).mean())
    return cross_entropy, brier, accuracy


def coalition_games(checkpoint, updates, weights, x, y) -> dict[str, np.ndarray]:
    n = len(updates)
    base_ce, base_brier, base_accuracy = utility(checkpoint, x, y)
    games = {name: np.zeros(1 << n) for name in ("negative_brier", "negative_cross_entropy", "accuracy")}
    for mask in range(1, 1 << n):
        ce, brier, accuracy = utility(add_weighted(checkpoint, updates, weights, mask), x, y)
        games["negative_brier"][mask] = base_brier - brier
        games["negative_cross_entropy"][mask] = base_ce - ce
        games["accuracy"][mask] = accuracy - base_accuracy
    return games


def partition(y: np.ndarray, clients: int, alpha: float, seed: int, minimum: int) -> list[np.ndarray]:
    for attempt in range(10000):
        rng = np.random.default_rng(seed + 104729 * attempt)
        buckets = [[] for _ in range(clients)]
        for label in np.unique(y):
            indices = rng.permutation(np.flatnonzero(y == label))
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
            cursor = 0
            for client, count in enumerate(counts):
                buckets[client].extend(indices[cursor : cursor + count])
                cursor += count
        if min(map(len, buckets)) >= minimum:
            return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]
    raise RuntimeError("could not satisfy minimum client size")


def mobius(values: np.ndarray, n: int) -> np.ndarray:
    result = values.copy()
    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit):
                result[mask] -= result[mask ^ (1 << bit)]
    return result


def analyze(values: np.ndarray, n: int) -> tuple[dict, list[dict], np.ndarray]:
    dividends = mobius(values, n)
    shapley = np.zeros(n)
    predicted = np.zeros(n)
    for mask in range(1, 1 << n):
        size = mask.bit_count()
        for i in range(n):
            if mask & (1 << i):
                shapley[i] += dividends[mask] / size
                if size >= 3:
                    predicted[i] += (0.5 - 1 / size) * dividends[mask]
    singleton = np.asarray([values[1 << i] for i in range(n)])
    grand = (1 << n) - 1
    loo = np.asarray([values[grand] - values[grand ^ (1 << i)] for i in range(n)])
    midpoint = (singleton + loo) / 2
    observed = midpoint - shapley
    tau = float(kendalltau(midpoint, shapley).statistic)
    if math.isnan(tau):
        tau = float(np.allclose(midpoint, shapley))
    metrics = {
        "high_order_mass": float(sum(abs(dividends[m]) for m in range(1, 1 << n) if m.bit_count() >= 3) / (np.abs(dividends[1:]).sum() + 1e-12)),
        "relative_midpoint_gap": float(np.abs(observed).sum() / (np.abs(shapley).sum() + 1e-12)),
        "absolute_midpoint_gap": float(np.abs(observed).sum()),
        "kendall_tau": tau,
        "sign_agreement": float(np.mean(np.signbit(midpoint) == np.signbit(shapley))),
        "efficiency_defect": float(midpoint.sum() - values[-1]),
        "identity_residual": float(np.max(np.abs(observed - predicted))),
    }
    clients = [{"client": i, "shapley": shapley[i], "midpoint": midpoint[i], "observed_error": observed[i], "predicted_error": predicted[i]} for i in range(n)]
    return metrics, clients, dividends


def bootstrap(rows: list[dict], metric: str, repeats: int, seed: int) -> dict[str, float]:
    clusters = defaultdict(list)
    for row in rows:
        clusters[int(row["partition_seed"])].append(float(row[metric]))
    ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(ids, len(ids), replace=True)
        estimates.append(np.median([value for key in sampled for value in clusters[int(key)]]))
    values = [value for key in ids for value in clusters[key]]
    return {"median": float(np.median(values)), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975))}


def scaling_slopes(
    rows: list[dict], utility_name: str, fit_max_scale: float, gap_floor: float
) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        if row["utility"] == utility_name and float(row["update_scale"]) <= fit_max_scale:
            key = (
                row["alpha"],
                row["partition_seed"],
                row["checkpoint_epoch"],
                row["local_epochs"],
            )
            groups[key].append(row)
    result, inconclusive = [], []
    for (alpha, seed, checkpoint, local_epochs), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: float(row["update_scale"]))
        scales = np.asarray([float(row["update_scale"]) for row in group])
        absolute = np.asarray([float(row["absolute_midpoint_gap"]) for row in group])
        relative = np.asarray([float(row["relative_midpoint_gap"]) for row in group])
        if len(group) < 3 or np.any(absolute <= gap_floor) or np.any(relative <= gap_floor):
            inconclusive.append(
                {
                    "alpha": alpha,
                    "partition_seed": seed,
                    "checkpoint_epoch": checkpoint,
                    "local_epochs": local_epochs,
                    "reason": "fewer than three fitted scales or a gap at the numerical floor",
                    "minimum_absolute_gap": float(absolute.min()),
                    "minimum_relative_gap": float(relative.min()),
                }
            )
            continue
        result.append(
            {
                "alpha": alpha,
                "partition_seed": seed,
                "checkpoint_epoch": checkpoint,
                "local_epochs": local_epochs,
                "absolute_log_slope": float(np.polyfit(np.log(scales), np.log(absolute), 1)[0]),
                "relative_log_slope": float(np.polyfit(np.log(scales), np.log(relative), 1)[0]),
                "minimum_absolute_gap": float(absolute.min()),
                "maximum_identity_residual": float(max(row["identity_residual"] for row in group)),
            }
        )
    return result, inconclusive


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parsed = args()
    config = json.loads(parsed.config.read_text())
    expected = json.loads((Path(__file__).parent / "input-hashes.json").read_text())
    for name, digest in expected.items():
        if sha256(parsed.data / name) != digest:
            raise ValueError(f"input hash mismatch: {name}")
    output = parsed.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    x_train, y_train, x_test, y_test = prepare(parsed.data, config)
    n = int(config["clients"])
    initial = initialize(x_train.shape[1], int(config["hidden_units"]), 10, int(config["initialization_seed"]))
    checkpoints = {}
    current, prior = initial, 0
    for epoch in sorted(config["global_epochs"]):
        current = train(current, x_train, y_train, int(epoch) - prior, config["global_learning_rate"], config["l2"])
        checkpoints[int(epoch)] = tuple(value.copy() for value in current)
        prior = int(epoch)

    round_rows, client_rows, dividend_rows = [], [], []
    game_id = 0
    scaling_mode = "update_scales" in config
    update_scales = [float(value) for value in config.get("update_scales", [1.0])]
    for alpha in config["dirichlet_alphas"]:
        for seed in config["partition_seeds"]:
            groups = partition(y_train, n, float(alpha), int(seed), int(config["minimum_client_examples"]))
            weights = np.asarray([len(group) for group in groups], dtype=float)
            weights /= weights.sum()
            for checkpoint_epoch, checkpoint in checkpoints.items():
                for local_epochs in config["local_epochs"]:
                    updates = [subtract(train(checkpoint, x_train[group], y_train[group], int(local_epochs), config["local_learning_rate"], config["l2"]), checkpoint) for group in groups]
                    base_norm_sum, base_norm_max, base_aggregate_norm = weighted_update_norms(updates, weights)
                    for update_scale in update_scales:
                        scaled_updates = [scale_update(update, update_scale) for update in updates]
                        for name, values in coalition_games(checkpoint, scaled_updates, weights, x_test, y_test).items():
                            metrics, clients, dividends = analyze(values, n)
                            if metrics["identity_residual"] > config["identity_tolerance"]:
                                raise AssertionError("Harsanyi identity residual exceeds tolerance")
                            common = {"game_id": game_id, "utility": name, "alpha": alpha, "partition_seed": seed, "checkpoint_epoch": checkpoint_epoch, "local_epochs": local_epochs}
                            if scaling_mode:
                                common.update(
                                    {
                                        "update_scale": update_scale,
                                        "base_weighted_update_norm_sum": base_norm_sum,
                                        "base_max_weighted_update_norm": base_norm_max,
                                        "base_weighted_aggregate_norm": base_aggregate_norm,
                                        "scaled_weighted_update_norm_sum": update_scale * base_norm_sum,
                                    }
                                )
                            round_rows.append({**common, **metrics})
                            client_rows.extend({**common, **row} for row in clients)
                            dividend_rows.extend({**common, "coalition_mask": mask, "coalition_size": mask.bit_count(), "dividend": dividends[mask]} for mask in range(1, 1 << n))
                            game_id += 1

    summary = {"schema_version": 1, "games": len(round_rows), "identity_residual_max": max(row["identity_residual"] for row in round_rows), "by_utility_alpha": {}, "by_regime": {}}
    for utility_name in sorted({row["utility"] for row in round_rows}):
        for alpha in config["dirichlet_alphas"]:
            subset = [row for row in round_rows if row["utility"] == utility_name and row["alpha"] == alpha]
            summary["by_utility_alpha"][f"{utility_name}|{alpha}"] = {metric: bootstrap(subset, metric, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + offset) for offset, metric in enumerate(("high_order_mass", "relative_midpoint_gap", "absolute_midpoint_gap", "kendall_tau"))}
            for checkpoint_epoch in config["global_epochs"]:
                for local_epochs in config["local_epochs"]:
                    for update_scale in update_scales:
                        regime = [row for row in subset if row["checkpoint_epoch"] == checkpoint_epoch and row["local_epochs"] == local_epochs and (not scaling_mode or row["update_scale"] == update_scale)]
                        key = f"{utility_name}|{alpha}|checkpoint={checkpoint_epoch}|local={local_epochs}"
                        if scaling_mode:
                            key += f"|scale={update_scale:g}"
                        summary["by_regime"][key] = {metric: bootstrap(regime, metric, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + 100 + offset) for offset, metric in enumerate(("high_order_mass", "relative_midpoint_gap", "absolute_midpoint_gap", "kendall_tau"))}
    for utility_name in sorted({row["utility"] for row in round_rows}):
        subset = [row for row in round_rows if row["utility"] == utility_name]
        summary.setdefault("spearman", {})[utility_name] = float(spearmanr([row["high_order_mass"] for row in subset], [row["relative_midpoint_gap"] for row in subset]).statistic)
    primary_regimes = {
        key: value
        for key, value in summary["by_regime"].items()
        if key.startswith("negative_brier|")
    }
    summary["material_primary_regimes"] = [
        key
        for key, value in primary_regimes.items()
        if value["high_order_mass"]["ci_low"] > config["material_high_order_mass"]
        and value["relative_midpoint_gap"]["median"] > config["material_relative_gap"]
    ]

    slope_rows, inconclusive_slope_rows = [], []
    if scaling_mode:
        slope_rows, inconclusive_slope_rows = scaling_slopes(
            round_rows,
            str(config["scaling_primary_utility"]),
            float(config["scaling_fit_max_scale"]),
            float(config["scaling_gap_floor"]),
        )
        if not slope_rows:
            raise RuntimeError("all scaling units are numerically inconclusive")
        summary["scaling"] = {
            "primary_utility": config["scaling_primary_utility"],
            "fit_max_scale": config["scaling_fit_max_scale"],
            "units": len(slope_rows),
            "inconclusive_units": len(inconclusive_slope_rows),
            "absolute_log_slope": bootstrap(
                slope_rows, "absolute_log_slope", int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + 500
            ),
            "relative_log_slope": bootstrap(
                slope_rows, "relative_log_slope", int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + 501
            ),
        }
        absolute_interval = config["absolute_slope_support_interval"]
        relative_interval = config["relative_slope_support_interval"]
        summary["scaling"]["support_gate"] = bool(
            absolute_interval[0] <= summary["scaling"]["absolute_log_slope"]["median"] <= absolute_interval[1]
            and relative_interval[0] <= summary["scaling"]["relative_log_slope"]["median"] <= relative_interval[1]
        )
        summary["scaling"]["strong_support_gate"] = bool(
            not inconclusive_slope_rows
            and absolute_interval[0] <= summary["scaling"]["absolute_log_slope"]["ci_low"]
            and summary["scaling"]["absolute_log_slope"]["ci_high"] <= absolute_interval[1]
            and relative_interval[0] <= summary["scaling"]["relative_log_slope"]["ci_low"]
            and summary["scaling"]["relative_log_slope"]["ci_high"] <= relative_interval[1]
        )

    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_csv(output / "rounds.csv", round_rows)
    write_csv(output / "clients.csv", client_rows)
    if slope_rows:
        write_csv(output / "scaling_units.csv", slope_rows)
    if inconclusive_slope_rows:
        write_csv(output / "scaling_inconclusive.csv", inconclusive_slope_rows)
    with gzip.open(output / "dividends.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dividend_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(dividend_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    checksums = [f"{sha256(path)}  {path.name}" for path in sorted(output.iterdir()) if path.is_file()]
    (output / "checksums.txt").write_text("\n".join(checksums) + "\n")
    if sum(path.stat().st_size for path in output.iterdir()) > config["max_output_bytes"]:
        raise RuntimeError("output limit exceeded")


if __name__ == "__main__":
    main()
