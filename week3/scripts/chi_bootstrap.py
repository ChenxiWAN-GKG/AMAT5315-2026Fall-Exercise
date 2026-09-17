#!/usr/bin/env python3
"""Block-bootstrap susceptibility peak fits for the Metropolis window runs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLOCK_LENGTHS = (2000, 4000, 8000)
REPLICATES = 500
EXACT_TC = 2.26919


def read_window(path: Path, expected_side: int) -> dict[float, np.ndarray]:
    with (path / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["update"] != "metropolis":
        raise ValueError(f"{path}: expected a Metropolis run")
    if int(metadata["L"]) != expected_side:
        raise ValueError(f"{path}: expected L={expected_side}")

    values: dict[float, list[float]] = {}
    with (path / "series.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            values.setdefault(float(record["T"]), []).append(float(record["M"]))

    expected_measure = int(metadata["measure"])
    result = {}
    for temperature, samples in sorted(values.items()):
        if len(samples) != expected_measure:
            raise ValueError(
                f"{path}: T={temperature:g} has {len(samples)} rows, "
                f"expected {expected_measure}"
            )
        result[temperature] = np.asarray(samples, dtype=np.float64)
    if not result:
        raise ValueError(f"{path}: no measured rows")
    return result


def susceptibility(samples: np.ndarray, temperature: float, side: int) -> float:
    mean_abs_m = np.mean(np.abs(samples))
    mean_m2 = np.mean(samples * samples)
    return side * side * (mean_m2 - mean_abs_m * mean_abs_m) / temperature


def fit_peak(temperatures: np.ndarray, susceptibilities: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit five points around the largest susceptibility and return vertex and coefficients."""
    peak_index = int(np.argmax(susceptibilities))
    if peak_index < 2 or peak_index + 2 >= len(temperatures):
        raise ValueError("largest susceptibility is too close to a temperature-grid boundary")

    fit_temperatures = temperatures[peak_index - 2 : peak_index + 3]
    fit_values = susceptibilities[peak_index - 2 : peak_index + 3]
    coefficients = np.polyfit(fit_temperatures, fit_values, 2)
    curvature, linear, _ = coefficients
    if curvature >= 0.0:
        raise ValueError("susceptibility fit does not bend downward")
    peak_temperature = -linear / (2.0 * curvature)
    if not fit_temperatures[0] <= peak_temperature <= fit_temperatures[-1]:
        raise ValueError("fitted peak is outside the five fitted temperatures")
    return float(peak_temperature), coefficients


def block_statistics(samples: np.ndarray, block_length: int) -> np.ndarray:
    block_count = len(samples) // block_length
    if block_count < 2:
        raise ValueError(f"{len(samples)} samples cannot form enough blocks of length {block_length}")
    blocks = samples[: block_count * block_length].reshape(block_count, block_length)
    return np.column_stack(
        (
            np.sum(blocks, axis=1),
            np.sum(blocks * blocks, axis=1),
            np.sum(np.abs(blocks), axis=1),
        )
    )


def bootstrap_fits(
    data: dict[float, np.ndarray], side: int, block_length: int, seed: int
) -> tuple[np.ndarray, list[np.ndarray], int]:
    temperatures = np.asarray(sorted(data), dtype=np.float64)
    stats = [block_statistics(data[temperature], block_length) for temperature in temperatures]
    rng = np.random.default_rng(seed)
    replicate_count = stats[0].shape[0]
    indices = rng.integers(0, replicate_count, size=(REPLICATES, replicate_count))
    peak_temperatures = []
    coefficients = []
    failed = 0
    for replicate_indices in indices:
        chi_values = []
        for temperature, block_stat in zip(temperatures, stats):
            selected = block_stat[replicate_indices]
            count = block_length * len(selected)
            mean_m = np.sum(selected[:, 0]) / count
            mean_m2 = np.sum(selected[:, 1]) / count
            mean_abs_m = np.sum(selected[:, 2]) / count
            chi_values.append(side * side * (mean_m2 - mean_abs_m**2) / temperature)
        try:
            peak, fit_coefficients = fit_peak(temperatures, np.asarray(chi_values))
        except ValueError:
            failed += 1
            continue
        peak_temperatures.append(peak)
        coefficients.append(fit_coefficients)
    return np.asarray(peak_temperatures), coefficients, failed


def stable(errors: list[float]) -> bool:
    mean_error = float(np.mean(errors))
    return max(errors) - min(errors) <= 0.1 * mean_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "evidence" / "chi-bootstrap.png"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report_path = args.report or args.out.with_suffix(".txt")

    data = {
        32: read_window(args.artifacts / "window-l32", 32),
        64: read_window(args.artifacts / "window-l64", 64),
    }
    colors = {2000: "tab:blue", 4000: "tab:orange", 8000: "tab:green"}
    results: dict[int, dict[int, tuple[np.ndarray, list[np.ndarray], int]]] = {32: {}, 64: {}}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for axis, side in zip(axes, (32, 64)):
        temperatures = np.asarray(sorted(data[side]), dtype=np.float64)
        chi_values = np.asarray(
            [susceptibility(data[side][temperature], temperature, side) for temperature in temperatures]
        )
        central_peak, central_coefficients = fit_peak(temperatures, chi_values)
        peak_index = int(np.argmax(chi_values))
        fit_range = np.linspace(temperatures[peak_index - 2], temperatures[peak_index + 2], 300)
        axis.plot(temperatures, chi_values, "ko", ms=3, label="window data")
        axis.plot(fit_range, np.polyval(central_coefficients, fit_range), "k-", label="five-point fit")
        axis.axvline(EXACT_TC, color="0.35", linestyle="--", label="exact $T_c$" if side == 32 else None)
        for block_length in BLOCK_LENGTHS:
            peaks, coefficients, failed = bootstrap_fits(
                data[side], side, block_length, args.seed + side + block_length
            )
            results[side][block_length] = (peaks, coefficients, failed)
            grid = np.linspace(temperatures[peak_index - 2], temperatures[peak_index + 2], 300)
            fitted_values = np.asarray([np.polyval(coefficient, grid) for coefficient in coefficients])
            axis.fill_between(
                grid,
                np.min(fitted_values, axis=0),
                np.max(fitted_values, axis=0),
                color=colors[block_length],
                alpha=0.18,
                label=f"{block_length}-move envelope",
            )
        axis.set_title(f"L = {side}, central fit peak = {central_peak:.4f}")
        axis.set_xlabel("Temperature T")
        axis.set_ylabel("Susceptibility χ")
        axis.grid(alpha=0.25)
        axis.set_xlim(2.0, 2.6)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)

    errors_by_length = {}
    report_lines = [
        "2D Ising susceptibility block-bootstrap",
        "500 replicates per block length; window runs only; observable is chi(T)",
        "Failed fit: upward-opening parabola or vertex outside its five fitted temperatures",
        "Stable means the three bootstrap errors agree within 10% of their mean",
        "",
    ]
    for block_length in BLOCK_LENGTHS:
        peaks32, _, failed32 = results[32][block_length]
        peaks64, _, failed64 = results[64][block_length]
        tc_peaks = 2.0 * peaks64 - peaks32
        error = float(np.std(tc_peaks, ddof=1))
        errors_by_length[block_length] = error
        report_lines.extend(
            [
                f"block_length={block_length} bootstrap_error_Tc={error:.8f}",
                f"  L=32 failed_fits={failed32} valid_fits={len(peaks32)}",
                f"  L=64 failed_fits={failed64} valid_fits={len(peaks64)}",
            ]
        )
    report_lines.append(
        f"stable_across_block_lengths={stable(list(errors_by_length.values()))}"
    )
    report = "\n".join(report_lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
