#!/usr/bin/env python3
"""Shared deterministic code for the bounded-logistic UCI-HAR experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


FEATURES = 561
CLASSES = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"input hash mismatch for {path}: {observed} != {expected}")


def verify_split(split_dir: Path, split: str, expected: dict[str, str]) -> None:
    suffix = "train" if split == "train" else "test"
    for stem in ("X", "y", "subject"):
        name = f"{stem}_{suffix}.txt"
        verify_hash(split_dir / name, expected[f"{split}/{name}"])


def load_split(split_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = "train" if split == "train" else "test"
    features = np.loadtxt(split_dir / f"X_{suffix}.txt", dtype=np.float64)
    labels = np.loadtxt(split_dir / f"y_{suffix}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(split_dir / f"subject_{suffix}.txt", dtype=np.int64)
    if features.shape != (len(labels), FEATURES) or len(subjects) != len(labels):
        raise ValueError(f"unexpected {split} shapes: {features.shape}, {labels.shape}, {subjects.shape}")
    return features, labels, subjects


def subject_roles(subjects: np.ndarray, seed: int, salt: str, clients: int) -> dict[str, list[int]]:
    unique = sorted(int(value) for value in np.unique(subjects))
    if len(unique) != 21:
        raise ValueError(f"expected 21 official-training subjects, got {len(unique)}")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{salt}{seed}|{value}".encode()).hexdigest(),
    )
    return {"checkpoint": ranked[:-clients], "players": ranked[-clients:]}


def fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def bounded_features(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Standardize, append a bias coordinate, and project each row into an L2 ball."""
    standardized = (features - mean) / scale
    augmented = np.column_stack([standardized, np.ones(len(features), dtype=np.float64)])
    norms = np.linalg.norm(augmented, axis=1, keepdims=True)
    denominators = np.maximum(1.0, norms / radius)
    result = augmented / denominators
    if float(np.max(np.linalg.norm(result, axis=1))) > radius * (1 + 1e-12):
        raise ArithmeticError("feature projection exceeded the declared radius")
    return result


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def mean_negative_cross_entropy(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    logits = x @ weights
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(shifted[np.arange(len(y)), y] - log_normalizer))


def train_full_batch(
    initial: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    result = initial.copy()
    targets = np.eye(result.shape[1], dtype=np.float64)[y]
    for _ in range(epochs):
        probabilities = stable_softmax(x @ result)
        gradient = x.T @ (probabilities - targets) / len(x)
        gradient[:-1] += l2 * result[:-1]
        result -= learning_rate * gradient
    return result


def clip_frobenius(update: np.ndarray, radius: float) -> tuple[np.ndarray, float, bool]:
    norm = float(np.linalg.norm(update))
    if norm <= radius:
        return update, norm, False
    return update * (radius / norm), norm, True


def coalition_values(
    checkpoint: np.ndarray,
    weighted_updates: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    n = len(weighted_updates)
    masks = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    base_logits = x @ checkpoint
    update_logits = np.einsum("ndc,xd->nxc", weighted_updates, x, optimize=True)
    logits = base_logits[None, :, :] + epsilon * np.einsum(
        "mn,nxc->mxc", masks, update_logits, optimize=True
    )
    shifted = logits - logits.max(axis=2, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=2))
    correct = np.take_along_axis(
        shifted, np.broadcast_to(y[None, :, None], (1 << n, len(y), 1)), axis=2
    )[:, :, 0]
    utilities = np.mean(correct - log_normalizer, axis=1)
    return utilities - utilities[0]


def scalar_radii(caps: list[float], epsilon: float, third_derivative_bound: float) -> list[float]:
    result = []
    for player, cap in enumerate(caps):
        others = [index for index in range(len(caps)) if index != player]
        pair_mass = sum(
            caps[others[left]] * caps[others[right]]
            for left in range(len(others))
            for right in range(left + 1, len(others))
        )
        result.append(third_derivative_bound * epsilon**3 * cap * pair_mass / 6.0)
    return result


def exact_shapley(values: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(n, dtype=np.float64)
    grand = (1 << n) - 1
    for player in range(n):
        bit = 1 << player
        for mask in range(grand + 1):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = 1.0 / (n * math.comb(n - 1, size))
            result[player] += weight * (values[mask | bit] - values[mask])
    return result


def midpoint(values: np.ndarray, n: int) -> np.ndarray:
    grand = (1 << n) - 1
    singleton = np.asarray([values[1 << player] for player in range(n)])
    leave_one_out = np.asarray([values[grand] - values[grand ^ (1 << player)] for player in range(n)])
    return 0.5 * (singleton + leave_one_out)


def write_checksums(directory: Path) -> None:
    rows = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "checksums.txt"
    ]
    (directory / "checksums.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
