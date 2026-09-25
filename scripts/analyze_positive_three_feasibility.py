#!/usr/bin/env python3
"""Reproduce the paper's positive third-order scope check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "inputs" / "positive_three_endpoints.json"
DEFAULT_OUTPUT = ROOT / "reproduced" / "positive-three-feasibility"
TOLERANCE_MULTIPLIER = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_three_interval(
    singleton: list[float], grand_marginal: list[float], grand_worth: float
) -> dict[str, object]:
    residual = [marginal - single for single, marginal in zip(singleton, grand_marginal)]
    interaction_surplus = grand_worth - sum(singleton)
    residual_sum = sum(residual)
    pair_mass = 3 * interaction_surplus - residual_sum
    triple_mass = residual_sum - 2 * interaction_surplus
    scale = max(
        1.0,
        abs(grand_worth),
        abs(pair_mass),
        abs(triple_mass),
        *(abs(value) for value in singleton + grand_marginal + residual),
    )
    tolerance = TOLERANCE_MULTIPLIER * scale
    lower_box = [max(0.0, value - pair_mass) for value in residual]
    upper_box = [min(triple_mass, value) for value in residual]
    conditions = {
        "nonnegative_singletons": min(singleton) >= -tolerance,
        "nonnegative_pair_mass": pair_mass >= -tolerance,
        "nonnegative_triple_mass": triple_mass >= -tolerance,
        "nonempty_coordinate_intervals": all(
            lower <= upper + tolerance for lower, upper in zip(lower_box, upper_box)
        ),
        "slice_lower_bound": sum(lower_box) <= 3 * triple_mass + tolerance,
        "slice_upper_bound": 3 * triple_mass <= sum(upper_box) + tolerance,
    }
    result: dict[str, object] = {
        "feasible": all(conditions.values()),
        "tolerance": tolerance,
        "interaction_surplus": interaction_surplus,
        "pair_mass": pair_mass,
        "triple_mass": triple_mass,
        **conditions,
    }
    if not result["feasible"]:
        return result

    triple_lower = [
        max(
            lower_box[i],
            3 * triple_mass
            - sum(upper_box[j] for j in range(len(upper_box)) if j != i),
        )
        for i in range(len(singleton))
    ]
    triple_upper = [
        min(
            upper_box[i],
            3 * triple_mass
            - sum(lower_box[j] for j in range(len(lower_box)) if j != i),
        )
        for i in range(len(singleton))
    ]
    shapley_lower = [
        singleton[i] + residual[i] / 2 - triple_upper[i] / 6
        for i in range(len(singleton))
    ]
    shapley_upper = [
        singleton[i] + residual[i] / 2 - triple_lower[i] / 6
        for i in range(len(singleton))
    ]
    widths = [upper - lower for lower, upper in zip(shapley_lower, shapley_upper)]
    result.update(
        shapley_lower=shapley_lower,
        shapley_upper=shapley_upper,
        mean_player_width=statistics.mean(widths),
        max_player_width=max(widths),
        sum_player_width=sum(widths),
    )
    return result


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    feasible = [row for row in rows if row["feasible"]]
    condition_names = (
        "nonnegative_singletons",
        "nonnegative_pair_mass",
        "nonnegative_triple_mass",
        "nonempty_coordinate_intervals",
        "slice_lower_bound",
        "slice_upper_bound",
    )
    failures = Counter(
        name for row in rows for name in condition_names if not row[name]
    )
    result: dict[str, object] = {
        "games": len(rows),
        "feasible_games": len(feasible),
        "feasible_fraction": len(feasible) / len(rows) if rows else math.nan,
        "failed_condition_counts": dict(sorted(failures.items())),
    }
    if feasible:
        for field in (
            "mean_player_width",
            "max_player_width",
            "sum_player_width",
            "normalized_sum_width",
        ):
            values = [float(row[field]) for row in feasible]
            result[field] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        result["exact_shapley_fully_covered"] = sum(
            bool(row["exact_shapley_fully_covered"]) for row in feasible
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    games = payload["games"]
    if len(games) != 273:
        raise AssertionError(f"expected 273 retained games, found {len(games)}")

    output_rows: list[dict[str, object]] = []
    for game in games:
        grand_worth = sum(game["exact_shapley"])
        result = positive_three_interval(
            game["singleton"], game["grand_marginal"], grand_worth
        )
        if result["feasible"]:
            tolerance = float(result["tolerance"])
            result["exact_shapley_fully_covered"] = all(
                lower - tolerance <= value <= upper + tolerance
                for value, lower, upper in zip(
                    game["exact_shapley"],
                    result["shapley_lower"],
                    result["shapley_upper"],
                )
            )
            result["normalized_sum_width"] = float(result["sum_player_width"]) / (
                sum(abs(value) for value in game["exact_shapley"]) + 1e-12
            )
        output_rows.append(
            {
                "phase": game["phase"],
                "source": game["source"],
                "game_id": game["game_id"],
                "dataset": game["dataset"],
                "clients": game["clients"],
                "grand_worth": grand_worth,
                **{
                    key: value
                    for key, value in result.items()
                    if not isinstance(value, list)
                },
            }
        )

    summary = {
        "schema_version": 1,
        "analysis": "boundary-only positive 3-additive feasibility and sharp intervals",
        "tolerance_multiplier": TOLERANCE_MULTIPLIER,
        "input_sha256": sha256(args.input),
        "overall": summarize(output_rows),
        "by_phase": {
            phase: summarize([row for row in output_rows if row["phase"] == phase])
            for phase in sorted({str(row["phase"]) for row in output_rows})
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    fieldnames = sorted({key for row in output_rows for key in row})
    games_path = args.output / "games.csv"
    with games_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "checksums.txt").write_text(
        f"{sha256(games_path)}  games.csv\n{sha256(summary_path)}  summary.json\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
