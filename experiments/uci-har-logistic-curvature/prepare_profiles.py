#!/usr/bin/env python3
"""Prepare bounded-logistic checkpoints and clipped client updates without test access."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from common import (
    CLASSES,
    bounded_features,
    clip_frobenius,
    fit_normalizer,
    load_json,
    load_split,
    mean_negative_cross_entropy,
    sha256,
    subject_roles,
    train_full_batch,
    verify_split,
    write_checksums,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--config", default=HERE / "config.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    expected = load_json(ROOT / config["inputs"]["hashes"])
    verify_split(args.train_dir, "train", expected)
    features, labels, subjects = load_split(args.train_dir, "train")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile_rows = []
    authorized_rows = []
    for seed in config["role_seeds"]:
        roles = subject_roles(
            subjects,
            int(seed),
            config["role_salt"],
            int(config["clients"]),
        )
        checkpoint_mask = np.isin(subjects, roles["checkpoint"])
        mean, scale = fit_normalizer(features[checkpoint_mask])
        transformed = bounded_features(
            features, mean, scale, float(config["feature_radius"])
        )
        dimensions = transformed.shape[1]
        initial = np.zeros((dimensions, CLASSES), dtype=np.float64)
        checkpoint = train_full_batch(
            initial,
            transformed[checkpoint_mask],
            labels[checkpoint_mask],
            int(config["checkpoint"]["epochs"]),
            float(config["checkpoint"]["learning_rate"]),
            float(config["checkpoint"]["l2"]),
        )

        counts = [int(np.sum(subjects == subject)) for subject in roles["players"]]
        total = sum(counts)
        weighted_updates = []
        realized_local_norms = []
        clipped_flags = []
        caps = []
        for subject, count in zip(roles["players"], counts):
            mask = subjects == subject
            local = train_full_batch(
                checkpoint,
                transformed[mask],
                labels[mask],
                int(config["local"]["epochs"]),
                float(config["local"]["learning_rate"]),
                float(config["local"]["l2"]),
            )
            clipped, realized_norm, clipped_flag = clip_frobenius(
                local - checkpoint, float(config["local_update_clip"])
            )
            weight = count / total
            weighted_updates.append(weight * clipped)
            realized_local_norms.append(realized_norm)
            clipped_flags.append(clipped_flag)
            caps.append(weight * float(config["local_update_clip"]))

        profile_path = output / f"profile-{seed}.npz"
        np.savez_compressed(
            profile_path,
            seed=np.asarray(seed, dtype=np.int64),
            checkpoint=checkpoint,
            weighted_updates=np.stack(weighted_updates),
            mean=mean,
            scale=scale,
            player_subjects=np.asarray(roles["players"], dtype=np.int64),
            player_counts=np.asarray(counts, dtype=np.int64),
        )
        profile_rows.append({
            "seed": int(seed),
            "path": profile_path.name,
            "sha256": sha256(profile_path),
            "checkpoint_negative_cross_entropy": mean_negative_cross_entropy(
                checkpoint, transformed[checkpoint_mask], labels[checkpoint_mask]
            ),
            "realized_local_update_norms": realized_local_norms,
            "clients_clipped": int(sum(clipped_flags)),
            "roles": roles,
        })
        authorized_rows.append({
            "seed": int(seed),
            "player_subjects": roles["players"],
            "player_counts": counts,
            "weighted_update_caps": caps,
        })
        print(f"prepared seed={seed} ({len(profile_rows)}/{len(config['role_seeds'])})", flush=True)

    authorized = {
        "schema_version": 1,
        "status": "public enforced clipping caps; no update coordinates",
        "config_sha256": sha256(args.config),
        "feature_radius": float(config["feature_radius"]),
        "local_update_clip": float(config["local_update_clip"]),
        "profiles": authorized_rows,
    }
    authorized_path = output / "authorized-caps.json"
    authorized_path.write_text(json.dumps(authorized, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "official_test_accessed": False,
        "all_profiles_prepared": len(profile_rows) == len(config["role_seeds"]),
        "config_sha256": sha256(args.config),
        "train_input_sha256": {
            key: value for key, value in expected.items() if key.startswith("train/")
        },
        "authorized_caps": {"path": authorized_path.name, "sha256": sha256(authorized_path)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
        "profiles": profile_rows,
    }
    (output / "profiles-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_checksums(output)


if __name__ == "__main__":
    main()
