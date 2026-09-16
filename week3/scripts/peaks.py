#!/usr/bin/env python3
"""Estimate susceptibility peaks and low-temperature magnetization."""

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np


LOWEST_TEMPERATURE = 1.5


def read_observables(path: Path, expected_side: int) -> dict[float, tuple[float, float]]:
    with (path / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["update"] != "metropolis":
        raise ValueError(f"{path}: expected a Metropolis run")
    if int(metadata["L"]) != expected_side:
        raise ValueError(f"{path}: expected L={expected_side}")

    expected_measure = int(metadata["measure"])
    magnetizations: dict[float, list[float]] = {}
    with (path / "series.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            temperature = float(record["T"])
            magnetizations.setdefault(temperature, []).append(float(record["M"]))

    result = {}
    for temperature, values in sorted(magnetizations.items()):
        if len(values) != expected_measure:
            raise ValueError(
                f"{path}: T={temperature:g} has {len(values)} rows, "
                f"expected {expected_measure}"
            )
        mean_abs_m = fmean(abs(value) for value in values)
        mean_m2 = fmean(value * value for value in values)
        chi = expected_side * expected_side * (mean_m2 - mean_abs_m**2) / temperature
        result[temperature] = (mean_abs_m, chi)
    if not result:
        raise ValueError(f"{path}: no measured rows")
    return result


def quadratic_peak(observables: dict[float, tuple[float, float]]) -> tuple[float, list[tuple[float, float]]]:
    temperatures = sorted(observables)
    peak_index = max(range(len(temperatures)), key=lambda i: observables[temperatures[i]][1])
    if peak_index < 2 or peak_index + 2 >= len(temperatures):
        raise ValueError("largest susceptibility is too close to a temperature-grid boundary")

    fit_temperatures = temperatures[peak_index - 2 : peak_index + 3]
    fit_values = [observables[temperature][1] for temperature in fit_temperatures]
    center = temperatures[peak_index]
    coefficients = np.polyfit(
        np.asarray(fit_temperatures) - center,
        np.asarray(fit_values),
        2,
    )
    curvature, linear, _ = coefficients
    if curvature >= 0:
        raise ValueError("susceptibility fit does not have a maximum")
    peak_temperature = center - linear / (2.0 * curvature)
    return float(peak_temperature), list(zip(fit_temperatures, fit_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "peaks.txt",
    )
    args = parser.parse_args()

    window_l32 = read_observables(args.artifacts / "window-l32", 32)
    window_l64 = read_observables(args.artifacts / "window-l64", 64)
    coarse_l32 = read_observables(args.artifacts / "coarse-l32", 32)
    coarse_l64 = read_observables(args.artifacts / "coarse-l64", 64)

    peak_l32, fit_l32 = quadratic_peak(window_l32)
    peak_l64, fit_l64 = quadratic_peak(window_l64)
    estimated_tc = 2.0 * peak_l64 - peak_l32
    if LOWEST_TEMPERATURE not in coarse_l32 or LOWEST_TEMPERATURE not in coarse_l64:
        raise ValueError(f"coarse runs must include T={LOWEST_TEMPERATURE:.1f}")

    lines = [
        "2D Ising susceptibility peak estimates",
        "chi(T) = L^2 (mean(M^2) - mean(|M|)^2) / T",
        "peak fits: window-l32 and window-l64, five points around each discrete maximum",
        f"T_peak(32) = {peak_l32:.8f}",
        f"T_peak(64) = {peak_l64:.8f}",
        f"T_c = 2*T_peak(64) - T_peak(32) = {estimated_tc:.8f}",
        f"mean |M| (L=32, T={LOWEST_TEMPERATURE:.2f}) = {coarse_l32[LOWEST_TEMPERATURE][0]:.8f}",
        f"mean |M| (L=64, T={LOWEST_TEMPERATURE:.2f}) = {coarse_l64[LOWEST_TEMPERATURE][0]:.8f}",
        "",
        "L=32 fit points:",
    ]
    lines.extend(f"  T={temperature:.2f}, chi={chi:.8f}" for temperature, chi in fit_l32)
    lines.append("L=64 fit points:")
    lines.extend(f"  T={temperature:.2f}, chi={chi:.8f}" for temperature, chi in fit_l64)
    output = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
