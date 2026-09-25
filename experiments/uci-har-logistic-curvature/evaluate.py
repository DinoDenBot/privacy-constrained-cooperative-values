#!/usr/bin/env python3
"""Evaluate frozen public-cap certificates against exact bounded-logistic games."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from common import (
    bounded_features,
    coalition_values,
    exact_shapley,
    load_json,
    load_split,
    midpoint,
    sha256,
    verify_hash,
    verify_split,
    write_checksums,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--certificates", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--config", default=HERE / "config.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_curvature():
    path = ROOT / "experiments/curvature-diagnostic/run.py"
    specification = importlib.util.spec_from_file_location("pcv_curvature", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load curvature diagnostic")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_profiles(directory: Path, config_path: Path, config: dict) -> dict:
    manifest = load_json(directory / "profiles-manifest.json")
    if not manifest.get("all_profiles_prepared") or manifest.get("official_test_accessed") is not False:
        raise ValueError("profiles were not sealed before official-test access")
    if manifest["config_sha256"] != sha256(config_path):
        raise ValueError("profile/config hash mismatch")
    if [int(row["seed"]) for row in manifest["profiles"]] != config["role_seeds"]:
        raise ValueError("profile seeds differ from the frozen design")
    for row in manifest["profiles"]:
        verify_hash(directory / row["path"], row["sha256"])
    caps = directory / manifest["authorized_caps"]["path"]
    verify_hash(caps, manifest["authorized_caps"]["sha256"])
    return manifest


def oracle_radius(curvature, values: np.ndarray, n: int, config: dict) -> np.ndarray:
    dividends = curvature.mobius_transform(values, n)
    result = []
    settings = config["curvature"]
    for player in range(n):
        bound, _ = curvature.bernstein_curvature_bound(
            curvature.rational_curvature_coefficients(dividends, n, player),
            float(settings["bernstein_relative_tolerance"]),
            float(settings["bernstein_absolute_tolerance"]),
            int(settings["bernstein_maximum_splits"]),
        )
        result.append(bound / 12.0 + float(settings["oracle_absolute_slack"]))
    return np.asarray(result)


def interval_result(
    curvature,
    truth: np.ndarray,
    estimate: np.ndarray,
    radius: np.ndarray,
    grand: float,
    config: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    lower = estimate - radius
    upper = estimate + radius
    efficient_lower, efficient_upper = curvature.tighten_by_efficiency(
        lower, upper, grand, float(config["identity_tolerance"])
    )
    result = curvature.interval_metrics(
        truth,
        efficient_lower,
        efficient_upper,
        grand,
        float(config["decision_tolerance"]),
        use_efficiency=True,
    )
    width = float((efficient_upper - efficient_lower).sum())
    result["efficient_l1_width"] = width
    result["efficient_relative_l1_width"] = width / (
        float(np.abs(truth).sum()) + float(config["relative_denominator_floor"])
    )
    return result, efficient_lower, efficient_upper


def summarize(rows: list[dict], epsilon: float, method: str) -> dict:
    selected = [
        row for row in rows
        if row["epsilon"] == epsilon and row["method"] == method
    ]
    players = sum(int(row["clients"]) for row in selected)
    strict_pairs = sum(int(row["strict_pairs"]) for row in selected)
    return {
        "games": len(selected),
        "positive_grand_worth_count": sum(row["grand_worth"] > 0 for row in selected),
        "joint_coverage_rate": float(np.mean([row["joint_coverage"] for row in selected])),
        "median_shapley_l1": float(np.median([row["shapley_l1"] for row in selected])),
        "median_midpoint_relative_l1_error": float(np.median([
            row["midpoint_relative_l1_error"] for row in selected
        ])),
        "median_efficient_relative_l1_width": float(np.median([
            row["efficient_relative_l1_width"] for row in selected
        ])),
        "sign_certification_rate": sum(row["signs_certified"] for row in selected) / players,
        "pair_certification_rate": (
            sum(row["pairs_certified"] for row in selected) / strict_pairs
            if strict_pairs else 1.0
        ),
        "top_client_certification_rate": float(np.mean([
            row["top_client_certified"] for row in selected
        ])),
        "median_radius_to_oracle_ratio": float(np.median([
            row["median_radius_to_oracle_ratio"] for row in selected
        ])),
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    manifest = verify_profiles(args.profiles, args.config, config)
    certificates_payload = load_json(args.certificates)
    expected_status = (
        "constructed from public clipping caps without update directions, test data, "
        "coalition values, or Shapley outcomes"
    )
    if certificates_payload.get("status") != expected_status:
        raise ValueError("certificate artifact has the wrong construction status")
    if certificates_payload["config_sha256"] != sha256(args.config):
        raise ValueError("certificate/config hash mismatch")
    certificates = {
        int(row["seed"]): row for row in certificates_payload["certificates"]
    }

    expected_hashes = load_json(ROOT / config["inputs"]["hashes"])
    verify_split(args.test_dir, "test", expected_hashes)
    test_features, test_labels, _ = load_split(args.test_dir, "test")
    curvature = load_curvature()
    n = int(config["clients"])
    game_rows = []
    player_rows = []
    exact_games = []
    for profile_row in manifest["profiles"]:
        seed = int(profile_row["seed"])
        payload = np.load(args.profiles / profile_row["path"])
        if int(payload["seed"]) != seed:
            raise ValueError("profile seed mismatch")
        x = bounded_features(
            test_features,
            payload["mean"],
            payload["scale"],
            float(config["feature_radius"]),
        )
        checkpoint = payload["checkpoint"]
        updates = payload["weighted_updates"]
        certificate = certificates[seed]
        if len(updates) != n or len(certificate["weighted_update_caps"]) != n:
            raise ValueError("unexpected client count")
        realized = np.linalg.norm(updates.reshape(n, -1), axis=1)
        caps = np.asarray(certificate["weighted_update_caps"], dtype=np.float64)
        if np.any(realized > caps * (1 + 1e-12)):
            raise AssertionError("realized weighted update exceeds its public cap")

        for epsilon in config["epsilons"]:
            epsilon = float(epsilon)
            values = coalition_values(
                checkpoint, updates, x, test_labels, epsilon
            )
            truth = exact_shapley(values, n)
            estimate = midpoint(values, n)
            grand = float(values[-1])
            if abs(float(truth.sum()) - grand) > config["identity_tolerance"]:
                raise AssertionError("Shapley efficiency identity failed")
            oracle = oracle_radius(curvature, values, n, config)
            methods = {
                "authorized_public_clip": np.asarray(
                    certificate["radii"][str(epsilon)], dtype=np.float64
                ),
                "oracle_bernstein": oracle,
            }
            exact_games.append({
                "seed": seed,
                "epsilon": epsilon,
                "player_subjects": payload["player_subjects"].tolist(),
                "coalition_values": values.tolist(),
                "shapley": truth.tolist(),
                "midpoint": estimate.tolist(),
                "grand_worth": grand,
            })
            shapley_l1 = float(np.abs(truth).sum())
            midpoint_error = float(np.abs(estimate - truth).sum())
            for method, radius in methods.items():
                result, lower, upper = interval_result(
                    curvature, truth, estimate, radius, grand, config
                )
                ratios = radius / np.maximum(oracle, float(config["curvature"]["oracle_absolute_slack"]))
                common = {
                    "seed": seed,
                    "epsilon": epsilon,
                    "method": method,
                    "clients": n,
                    "grand_worth": grand,
                    "shapley_l1": shapley_l1,
                    "midpoint_l1_error": midpoint_error,
                    "midpoint_relative_l1_error": midpoint_error / (
                        shapley_l1 + float(config["relative_denominator_floor"])
                    ),
                    "median_radius_to_oracle_ratio": float(np.median(ratios)),
                    **result,
                }
                game_rows.append(common)
                for player in range(n):
                    player_rows.append({
                        "seed": seed,
                        "epsilon": epsilon,
                        "method": method,
                        "player": player,
                        "subject": int(payload["player_subjects"][player]),
                        "exact_shapley": float(truth[player]),
                        "midpoint": float(estimate[player]),
                        "radius": float(radius[player]),
                        "efficient_lower": float(lower[player]),
                        "efficient_upper": float(upper[player]),
                        "oracle_bernstein_radius": float(oracle[player]),
                        "radius_to_oracle_ratio": float(ratios[player]),
                    })

    summaries = {
        str(float(epsilon)): {
            method: summarize(game_rows, float(epsilon), method)
            for method in ("authorized_public_clip", "oracle_bernstein")
        }
        for epsilon in config["epsilons"]
    }
    primary = summaries[str(float(config["primary_epsilon"]))]["authorized_public_clip"]
    thresholds = config["scientific_gates"]
    gates = {
        "coverage": primary["joint_coverage_rate"] >= thresholds["joint_coverage_rate_minimum"],
        "useful_games": primary["positive_grand_worth_count"] >= thresholds["positive_grand_worth_count_minimum"],
        "nontrivial_signal": primary["median_shapley_l1"] >= thresholds["median_shapley_l1_minimum"],
        "relative_width": primary["median_efficient_relative_l1_width"] < thresholds["median_efficient_relative_l1_width_strict_maximum"],
        "signs": primary["sign_certification_rate"] >= thresholds["sign_certification_rate_minimum"],
        "pairs": primary["pair_certification_rate"] >= thresholds["pair_certification_rate_minimum"],
        "top_client": primary["top_client_certification_rate"] >= thresholds["top_client_certification_rate_minimum"],
    }
    decision = {
        "schema_version": 1,
        "verdict": "supported" if all(gates.values()) else "contradicted",
        "all_primary_scientific_gates_passed": all(gates.values()),
        "primary_epsilon": float(config["primary_epsilon"]),
        "gates": gates,
        "thresholds": thresholds,
        "summaries": summaries,
        "certificate_input": {"path": str(args.certificates), "sha256": sha256(args.certificates)},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("games.csv", game_rows), ("players.csv", player_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (output / "exact-games.json").write_text(
        json.dumps({"schema_version": 1, "games": exact_games}, indent=2, sort_keys=True) + "\n"
    )
    (output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_checksums(output)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
