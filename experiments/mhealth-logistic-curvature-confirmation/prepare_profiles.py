#!/usr/bin/env python3
"""Prepare MHEALTH profiles without using each profile's held-out subject."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/uci-har-logistic-curvature"))
from common import bounded_features, clip_frobenius, fit_normalizer, sha256, train_full_batch, write_checksums  # noqa: E402


def load_subject(data_dir: Path, subject: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(data_dir / f"mHealth_subject{subject}.log", dtype=np.float64)
    if raw.shape[1] != 24:
        raise ValueError("MHEALTH row does not have 24 columns")
    labeled = raw[raw[:, -1] > 0][::stride]
    return labeled[:, :-1], labeled[:, -1].astype(np.int64) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--config", default=HERE / "config.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    hashes = json.loads((HERE / "input-hashes.json").read_text())
    for name, expected in hashes.items():
        if sha256(args.data_dir / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    subjects = {
        subject: load_subject(args.data_dir, subject, int(config["sample_stride"]))
        for subject in range(1, 11)
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile_rows, cap_rows = [], []
    for profile_id, roles in enumerate(config["profiles"]):
        checkpoint_subject = int(roles["checkpoint_subject"])
        checkpoint_x, checkpoint_y = subjects[checkpoint_subject]
        mean, scale = fit_normalizer(checkpoint_x)
        checkpoint_x = bounded_features(checkpoint_x, mean, scale, config["feature_radius"])
        checkpoint = train_full_batch(
            np.zeros((checkpoint_x.shape[1], int(config["classes"]))),
            checkpoint_x, checkpoint_y, **config["checkpoint"]
        )
        counts = [len(subjects[int(subject)][1]) for subject in roles["players"]]
        total = sum(counts)
        updates, caps, local_norms, clipped_flags = [], [], [], []
        for subject, count in zip(roles["players"], counts):
            local_x, local_y = subjects[int(subject)]
            local_x = bounded_features(local_x, mean, scale, config["feature_radius"])
            local = train_full_batch(checkpoint, local_x, local_y, **config["local"])
            clipped, local_norm, clipped_flag = clip_frobenius(
                local - checkpoint, config["local_update_clip"]
            )
            weight = count / total
            updates.append(weight * clipped)
            caps.append(weight * config["local_update_clip"])
            local_norms.append(local_norm)
            clipped_flags.append(clipped_flag)
        path = output / f"profile-{profile_id}.npz"
        np.savez_compressed(
            path, seed=profile_id, checkpoint=checkpoint,
            weighted_updates=np.stack(updates), mean=mean, scale=scale,
            player_subjects=np.asarray(roles["players"]),
            player_counts=np.asarray(counts), test_subject=int(roles["test_subject"])
        )
        profile_rows.append({
            "seed": profile_id, "path": path.name, "sha256": sha256(path),
            "checkpoint_subject": checkpoint_subject,
            "test_subject": int(roles["test_subject"]),
            "players": roles["players"], "clients_clipped": int(sum(clipped_flags)),
            "realized_local_update_norms": local_norms,
        })
        cap_rows.append({
            "seed": profile_id, "player_subjects": roles["players"],
            "player_counts": counts, "weighted_update_caps": caps,
        })
        print(f"prepared profile={profile_id} ({profile_id + 1}/{len(config['profiles'])})", flush=True)
    caps_payload = {
        "schema_version": 1, "status": "public enforced clipping caps; no update coordinates",
        "config_sha256": sha256(args.config), "feature_radius": config["feature_radius"],
        "local_update_clip": config["local_update_clip"], "profiles": cap_rows,
    }
    caps_path = output / "authorized-caps.json"
    caps_path.write_text(json.dumps(caps_payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "all_profiles_prepared": True,
        "designated_test_subject_used_in_profile_training": False,
        "config_sha256": sha256(args.config),
        "authorized_caps": {"path": caps_path.name, "sha256": sha256(caps_path)},
        "profiles": profile_rows,
    }
    (output / "profiles-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_checksums(output)


if __name__ == "__main__":
    main()
