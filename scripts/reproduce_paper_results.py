#!/usr/bin/env python3
"""Recompute and verify every empirical number reported in the paper."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "data"
TOLERANCE = 5e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(label: str, condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(f"{label}: {detail}")
    print(f"[PASS] {label}: {detail}")


def close(left: float, right: float, tolerance: float = TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def quantile(values: list[float], probability: float) -> float:
    """NumPy-compatible linear quantile for a sorted one-dimensional sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def verify_scaling() -> None:
    directory = RESULTS / "scaling-confirmatory-20260903"
    units = read_csv(directory / "scaling_units.csv")
    require("Fashion profile count", len(units) == 30, f"{len(units)} profiles")
    absolute = statistics.median(float(row["absolute_log_slope"]) for row in units)
    relative = statistics.median(float(row["relative_log_slope"]) for row in units)
    require("absolute scaling slope", close(absolute, 3.125392240492266), f"{absolute:.6f}")
    require("relative scaling slope", close(relative, 2.110000010670011), f"{relative:.6f}")

    raw = [
        row
        for row in read_csv(directory / "rounds.csv")
        if row["utility"] == "negative_brier"
    ]
    published = read_csv(FIGURES / "midpoint_scaling.csv")
    scales = sorted({float(row["update_scale"]) for row in raw})
    require("Fashion scale count", len(scales) == 5, f"{len(scales)} scales")
    require("Fashion figure row count", len(published) == len(scales), f"{len(published)} rows")
    anchors: dict[str, float] = {}
    for index, (scale, plotted) in enumerate(zip(scales, published)):
        selected = [row for row in raw if float(row["update_scale"]) == scale]
        require(f"Fashion observations at epsilon={scale:g}", len(selected) == 30, "30 games")
        expected = {"epsilon": scale}
        for short, field in (
            ("abs", "absolute_midpoint_gap"),
            ("rel", "relative_midpoint_gap"),
        ):
            values = [float(row[field]) for row in selected]
            expected[f"mid_{short}_lo"] = quantile(values, 0.25)
            expected[f"mid_{short}_med"] = quantile(values, 0.50)
            expected[f"mid_{short}_hi"] = quantile(values, 0.75)
            if index == 0:
                anchors[short] = expected[f"mid_{short}_med"]
            power = 3 if short == "abs" else 2
            expected[f"mid_{short}_ref"] = anchors[short] * (scale / scales[0]) ** power
        for field, value in expected.items():
            require(
                f"Fashion figure epsilon={scale:g} {field}",
                close(float(plotted[field]), value),
                f"{float(plotted[field]):.12g}",
            )


def certificate_summary(path: Path, epsilon: float) -> dict[str, float | int]:
    rows = [
        row
        for row in read_csv(path)
        if row["method"] == "authorized_public_clip"
        and close(float(row["epsilon"]), epsilon)
    ]
    return {
        "games": len(rows),
        "positive": sum(float(row["grand_worth"]) > 0 for row in rows),
        "width": statistics.median(float(row["efficient_relative_l1_width"]) for row in rows),
        "signs": sum(int(row["signs_certified"]) for row in rows),
        "pairs": sum(int(row["pairs_certified"]) for row in rows),
        "tops": sum(row["top_client_certified"] == "True" for row in rows),
        "gain_median": statistics.median(float(row["grand_worth"]) for row in rows),
        "gain_min": min(float(row["grand_worth"]) for row in rows),
        "gain_max": max(float(row["grand_worth"]) for row in rows),
    }


def verify_certificates() -> None:
    uci = certificate_summary(
        RESULTS / "uci-har-logistic-curvature-20260910" / "games.csv", 0.5
    )
    expected_uci = {"games": 20, "positive": 20, "signs": 160, "pairs": 539, "tops": 18}
    for key, expected in expected_uci.items():
        require(f"UCI-HAR {key}", uci[key] == expected, str(uci[key]))
    require("UCI-HAR median relative width", close(float(uci["width"]), 0.02619935286302015), "2.62%")

    mhealth = certificate_summary(
        RESULTS / "mhealth-logistic-curvature-confirmation-20260910" / "games.csv", 0.5
    )
    expected_mhealth = {"games": 20, "positive": 20, "signs": 159, "pairs": 555, "tops": 20}
    for key, expected in expected_mhealth.items():
        require(f"MHEALTH {key}", mhealth[key] == expected, str(mhealth[key]))
    require("MHEALTH median relative width", close(float(mhealth["width"]), 0.01240065137329667), "1.24%")
    require("MHEALTH median grand gain", close(float(mhealth["gain_median"]), 0.006063367075429871), f"{float(mhealth['gain_median']):.5f}")
    require("MHEALTH minimum grand gain", close(float(mhealth["gain_min"]), 0.0020709604518769353), f"{float(mhealth['gain_min']):.5f}")
    require("MHEALTH maximum grand gain", close(float(mhealth["gain_max"]), 0.013076943257481322), f"{float(mhealth['gain_max']):.5f}")

    players = [
        row
        for row in read_csv(
            RESULTS / "mhealth-logistic-curvature-confirmation-20260910" / "players.csv"
        )
        if row["profile"] == "0" and row["method"] == "authorized_public_clip"
    ]
    plotted = read_csv(FIGURES / "mhealth_certificate_example.csv")
    require("MHEALTH example client count", len(players) == len(plotted) == 8, "8 clients")
    for rank, (source, figure) in enumerate(zip(players, plotted), start=1):
        expected = {
            "subject": int(source["subject"]),
            "rank": rank,
            "efficient_lower": 1000 * float(source["efficient_lower"]),
            "midpoint": 1000 * float(source["midpoint"]),
            "exact_shapley": 1000 * float(source["exact_shapley"]),
            "efficient_upper": 1000 * float(source["efficient_upper"]),
            "error_minus": 1000 * (float(source["midpoint"]) - float(source["efficient_lower"])),
            "error_plus": 1000 * (float(source["efficient_upper"]) - float(source["midpoint"])),
        }
        for field, value in expected.items():
            observed = int(figure[field]) if field in {"subject", "rank"} else float(figure[field])
            require(
                f"MHEALTH example rank={rank} {field}",
                observed == value if field in {"subject", "rank"} else close(observed, value, 2e-10),
                str(observed),
            )


def verify_positive_three_scope() -> None:
    rows = [
        row
        for row in read_csv(RESULTS / "positive-three-feasibility-20260905" / "games.csv")
        if row["phase"] != "full_cifar_confirmation"
    ]
    feasible = [row for row in rows if row["feasible"] == "True"]
    covered = sum(row["exact_shapley_fully_covered"] == "True" for row in feasible)
    require("positive-third-order retained games", len(rows) == 273, f"{len(rows)} games")
    require("positive-third-order feasible observations", len(feasible) == 6, f"{len(feasible)} observations")
    require("positive-third-order feasible fraction", close(len(feasible) / len(rows), 6 / 273), f"{100 * len(feasible) / len(rows):.1f}%")
    require("positive-third-order fully covered exact vectors", covered == 0, f"{covered} of {len(feasible)}")


def main() -> None:
    verify_scaling()
    verify_certificates()
    verify_positive_three_scope()
    print("All paper-result checks passed.")


if __name__ == "__main__":
    main()
