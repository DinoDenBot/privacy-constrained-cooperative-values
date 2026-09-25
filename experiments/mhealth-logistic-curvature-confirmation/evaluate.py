#!/usr/bin/env python3
"""Open designated MHEALTH test subjects after certificate construction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UCI_BUNDLE = ROOT / "experiments/uci-har-logistic-curvature"
sys.path.insert(0, str(UCI_BUNDLE))
from common import bounded_features, coalition_values, exact_shapley, midpoint, sha256, write_checksums  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_EVALUATE = load_module("bounded_logistic_evaluate", UCI_BUNDLE / "evaluate.py")


def load_subject(data_dir: Path, subject: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(data_dir / f"mHealth_subject{subject}.log", dtype=np.float64)
    labeled = raw[raw[:, -1] > 0][::stride]
    return labeled[:, :-1], labeled[:, -1].astype(np.int64) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--certificates", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--config", default=HERE / "config.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    hashes = json.loads((HERE / "input-hashes.json").read_text())
    for name, expected in hashes.items():
        if sha256(args.data_dir / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    manifest = json.loads((args.profiles / "profiles-manifest.json").read_text())
    if manifest.get("designated_test_subject_used_in_profile_training") is not False:
        raise ValueError("profile construction did not preserve designated holdouts")
    if manifest["config_sha256"] != sha256(args.config):
        raise ValueError("profile/config hash mismatch")
    for row in manifest["profiles"]:
        if sha256(args.profiles / row["path"]) != row["sha256"]:
            raise ValueError("profile hash mismatch")
    certificates_payload = json.loads(args.certificates.read_text())
    if certificates_payload["config_sha256"] != sha256(args.config):
        raise ValueError("certificate/config hash mismatch")
    certificates = {int(row["seed"]): row for row in certificates_payload["certificates"]}
    curvature = BASE_EVALUATE.load_curvature()
    n = int(config["clients"])
    epsilon = float(config["primary_epsilon"])
    game_rows, player_rows, exact_games = [], [], []
    for profile_row in manifest["profiles"]:
        profile_id = int(profile_row["seed"])
        payload = np.load(args.profiles / profile_row["path"])
        test_subject = int(payload["test_subject"])
        raw_x, y = load_subject(args.data_dir, test_subject, int(config["sample_stride"]))
        x = bounded_features(raw_x, payload["mean"], payload["scale"], config["feature_radius"])
        updates = payload["weighted_updates"]
        certificate = certificates[profile_id]
        caps = np.asarray(certificate["weighted_update_caps"])
        realized = np.linalg.norm(updates.reshape(n, -1), axis=1)
        if np.any(realized > caps * (1 + 1e-12)):
            raise AssertionError("weighted update exceeds public cap")
        values = coalition_values(payload["checkpoint"], updates, x, y, epsilon)
        truth = exact_shapley(values, n)
        estimate = midpoint(values, n)
        grand = float(values[-1])
        oracle = BASE_EVALUATE.oracle_radius(curvature, values, n, config)
        exact_games.append({
            "profile": profile_id, "checkpoint_subject": profile_row["checkpoint_subject"],
            "test_subject": test_subject, "player_subjects": payload["player_subjects"].tolist(),
            "epsilon": epsilon, "coalition_values": values.tolist(), "shapley": truth.tolist(),
            "midpoint": estimate.tolist(), "grand_worth": grand,
        })
        for method, radius in {
            "authorized_public_clip": np.asarray(certificate["radii"][str(epsilon)]),
            "oracle_bernstein": oracle,
        }.items():
            metrics, lower, upper = BASE_EVALUATE.interval_result(
                curvature, truth, estimate, radius, grand, config
            )
            ratios = radius / np.maximum(oracle, config["curvature"]["oracle_absolute_slack"])
            shapley_l1 = float(np.abs(truth).sum())
            error = float(np.abs(estimate - truth).sum())
            game_rows.append({
                "seed": profile_id, "epsilon": epsilon, "method": method, "clients": n,
                "checkpoint_subject": profile_row["checkpoint_subject"], "test_subject": test_subject,
                "grand_worth": grand, "shapley_l1": shapley_l1,
                "midpoint_l1_error": error,
                "midpoint_relative_l1_error": error / (shapley_l1 + config["relative_denominator_floor"]),
                "median_radius_to_oracle_ratio": float(np.median(ratios)), **metrics,
            })
            for player in range(n):
                player_rows.append({
                    "profile": profile_id, "method": method, "player": player,
                    "subject": int(payload["player_subjects"][player]),
                    "exact_shapley": float(truth[player]), "midpoint": float(estimate[player]),
                    "radius": float(radius[player]), "efficient_lower": float(lower[player]),
                    "efficient_upper": float(upper[player]),
                    "oracle_bernstein_radius": float(oracle[player]),
                    "radius_to_oracle_ratio": float(ratios[player]),
                })
    summaries = {
        method: BASE_EVALUATE.summarize(game_rows, epsilon, method)
        for method in ("authorized_public_clip", "oracle_bernstein")
    }
    primary = summaries["authorized_public_clip"]
    threshold = config["scientific_gates"]
    gates = {
        "coverage": primary["joint_coverage_rate"] >= threshold["joint_coverage_rate_minimum"],
        "useful_games": primary["positive_grand_worth_count"] >= threshold["positive_grand_worth_count_minimum"],
        "nontrivial_signal": primary["median_shapley_l1"] >= threshold["median_shapley_l1_minimum"],
        "relative_width": primary["median_efficient_relative_l1_width"] < threshold["median_efficient_relative_l1_width_strict_maximum"],
        "signs": primary["sign_certification_rate"] >= threshold["sign_certification_rate_minimum"],
        "pairs": primary["pair_certification_rate"] >= threshold["pair_certification_rate_minimum"],
        "top_client": primary["top_client_certification_rate"] >= threshold["top_client_certification_rate_minimum"],
    }
    decision = {
        "schema_version": 1, "verdict": "supported" if all(gates.values()) else "contradicted",
        "all_scientific_gates_passed": all(gates.values()), "epsilon": epsilon,
        "gates": gates, "thresholds": threshold, "methods": summaries,
        "certificate_sha256": sha256(args.certificates),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("games.csv", game_rows), ("players.csv", player_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    (output / "exact-games.json").write_text(json.dumps({"games": exact_games}, indent=2, sort_keys=True) + "\n")
    (output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_checksums(output)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
