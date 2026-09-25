#!/usr/bin/env python3
"""Construct public-cap curvature certificates without reading update directions or games."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import load_json, scalar_radii, sha256, verify_hash, write_checksums


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", required=True, type=Path)
    parser.add_argument("--config", default=HERE / "config.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    caps_payload = load_json(args.caps)
    if caps_payload.get("status") != "public enforced clipping caps; no update coordinates":
        raise ValueError("caps artifact has the wrong information status")
    if caps_payload["config_sha256"] != sha256(args.config):
        raise ValueError("caps/config hash mismatch")
    if caps_payload["feature_radius"] != config["feature_radius"]:
        raise ValueError("feature radius mismatch")

    # For h_y(z)=log softmax(z)_y, D^3 h is the negative third joint
    # central moment of categorical logit directions. A categorical covariance
    # has spectral norm at most 1/2, and a centered unit direction has magnitude
    # at most sqrt(2). Cauchy--Schwarz therefore gives 1/sqrt(2). The linear
    # parameter-to-logit map contributes at most ||x||^3.
    logit_third_bound = 1.0 / math.sqrt(2.0)
    parameter_third_bound = logit_third_bound * float(config["feature_radius"]) ** 3
    certificates = []
    for row in caps_payload["profiles"]:
        caps = [float(value) for value in row["weighted_update_caps"]]
        certificates.append({
            "seed": int(row["seed"]),
            "weighted_update_caps": caps,
            "radii": {
                str(epsilon): scalar_radii(caps, float(epsilon), parameter_third_bound)
                for epsilon in config["epsilons"]
            },
        })

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "status": "constructed from public clipping caps without update directions, test data, coalition values, or Shapley outcomes",
        "caps_input": {"path": str(args.caps), "sha256": sha256(args.caps)},
        "config_sha256": sha256(args.config),
        "logit_third_derivative_bound": logit_third_bound,
        "parameter_third_derivative_bound": parameter_third_bound,
        "certificates": certificates,
    }
    (output / "certificates.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_checksums(output)


if __name__ == "__main__":
    main()
