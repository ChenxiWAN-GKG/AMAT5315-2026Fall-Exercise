#!/usr/bin/env python3
"""Compare Metropolis and Wolff measurements from the critical window."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.chi_bootstrap import bootstrap_fits, fit_peak
from scripts.errors import autocorrelation_time, combine_runs, read_run


EXACT_TC = 2.26919
BLOCK_LENGTHS = (2000, 4000, 8000)


def read_window(
    path: Path, side: int, update: str
) -> dict[float, tuple[np.ndarray, np.ndarray | None]]:
    """Read the measured magnetization and optional cluster size at each temperature."""
    with (path / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["L"] != side or metadata["update"] != update:
        raise ValueError(f"{path}: expected {update} L={side}")

    magnetizations: dict[float, list[float]] = {}
    clusters: dict[float, list[int]] = {}
    with (path / "series.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            temperature = float(row["T"])
            magnetizations.setdefault(temperature, []).append(float(row["M"]))
            if update == "wolff":
                clusters.setdefault(temperature, []).append(int(row["cluster_size"]))

    result = {}
    for temperature, values in sorted(magnetizations.items()):
        if len(values) != metadata["measure"]:
            raise ValueError(f"{path}: incomplete series at T={temperature:g}")
        cluster_sizes = (
            np.asarray(clusters[temperature], dtype=np.float64)
            if update == "wolff"
            else None
        )
        result[temperature] = (np.asarray(values, dtype=np.float64), cluster_sizes)
    return result


def work_normalized_tau(tau_moves: float, mean_cluster_size: float, side: int) -> float:
    """Express a cluster correlation time in L² spin updates."""
    return tau_moves * mean_cluster_size / (side * side)


def bootstrap_mean_error(values: np.ndarray, block_length: int, seed: int) -> float:
    """Resample contiguous blocks to estimate the mean's standard error."""
    block_count = len(values) // block_length
    blocks = values[: block_count * block_length].reshape(block_count, block_length)
    block_means = blocks.mean(axis=1)
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, block_count, size=(500, block_count))
    return float(np.std(block_means[selections].mean(axis=1), ddof=1))


def susceptibility(values: np.ndarray, temperature: float, side: int) -> float:
    mean_abs = np.mean(np.abs(values))
    return float(side * side * (np.mean(values * values) - mean_abs**2) / temperature)


def plot_magnetization(
    metropolis: dict[int, dict[float, tuple[np.ndarray, np.ndarray | None]]],
    wolff: dict[int, dict[float, tuple[np.ndarray, np.ndarray | None]]],
    output: Path,
) -> tuple[dict[int, float], float]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    comparisons = {}
    temperatures = sorted(metropolis[64])
    for update, data, color in (
        ("Metropolis", metropolis[64], "tab:orange"),
        ("Wolff", wolff[64], "tab:blue"),
    ):
        means = [np.mean(np.abs(data[t][0])) for t in temperatures]
        errors = [
            bootstrap_mean_error(np.abs(data[t][0]), 8000, 2026 + index)
            for index, t in enumerate(temperatures)
        ]
        axes[0].errorbar(temperatures, means, yerr=errors, marker="o", ms=3, capsize=2,
                         label=update, color=color)
        comparisons[update] = (means[temperatures.index(2.3)], errors[temperatures.index(2.3)])
    distance = abs(comparisons["Wolff"][0] - comparisons["Metropolis"][0])
    combined_error = np.hypot(comparisons["Wolff"][1], comparisons["Metropolis"][1])
    standardized_distance = distance / combined_error
    axes[0].axvline(EXACT_TC, color="0.4", ls="--", label="exact $T_c$")
    axes[0].set(xlabel="Temperature T", ylabel=r"mean $|M|$", title="L=64 magnetization")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    peaks = {}
    for side, color in ((32, "tab:green"), (64, "tab:blue")):
        grid = np.asarray(sorted(wolff[side]), dtype=np.float64)
        chi = np.asarray([susceptibility(wolff[side][t][0], t, side) for t in grid])
        peak, coefficients = fit_peak(grid, chi)
        peaks[side] = peak
        axes[1].plot(grid, chi, "o", ms=3, color=color, label=f"L={side}, peak {peak:.4f}")
        peak_index = int(np.argmax(chi))
        fitted_grid = np.linspace(grid[peak_index - 2], grid[peak_index + 2], 100)
        axes[1].plot(fitted_grid, np.polyval(coefficients, fitted_grid), color=color)
    tc_estimate = 2.0 * peaks[64] - peaks[32]
    axes[1].axvline(EXACT_TC, color="0.4", ls="--", label="exact $T_c$")
    axes[1].set(xlabel="Temperature T", ylabel="Susceptibility χ",
                title=f"Wolff five-point peaks; extrapolated $T_c$={tc_estimate:.4f}")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return peaks, standardized_distance


def plot_work(
    metropolis: dict[float, tuple[np.ndarray, np.ndarray | None]],
    wolff: dict[float, tuple[np.ndarray, np.ndarray | None]],
    output: Path,
) -> float:
    temperatures = sorted(metropolis)
    metropolis_tau = [autocorrelation_time(np.abs(metropolis[t][0])) for t in temperatures]
    wolff_tau = [
        work_normalized_tau(
            autocorrelation_time(np.abs(wolff[t][0])), float(np.mean(wolff[t][1])), 64
        )
        for t in temperatures
    ]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.semilogy(temperatures, metropolis_tau, "o-", ms=3, label="Metropolis")
    axis.semilogy(temperatures, wolff_tau, "o-", ms=3, label="Wolff")
    axis.axvline(EXACT_TC, color="0.4", ls="--", label="exact $T_c$")
    axis.set(xlabel="Temperature T", ylabel=r"$\tau_{\mathrm{work}}$ ($L^2$ spin updates)",
             title="Work per independent sample, L=64")
    axis.legend()
    axis.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    index = temperatures.index(2.3)
    return metropolis_tau[index] / wolff_tau[index]


def plot_metropolis_tau(
    metropolis: dict[int, dict[float, np.ndarray]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for side in (32, 64):
        temperatures = sorted(metropolis[side])
        taus = [autocorrelation_time(metropolis[side][t]) for t in temperatures]
        axis.semilogy(temperatures, taus, "o-", ms=3, label=f"L={side}")
    axis.axvline(EXACT_TC, color="0.4", ls="--", label="exact $T_c$")
    axis.set(xlabel="Temperature T", ylabel="Integrated autocorrelation time (sweeps)",
             title="Metropolis slowing near the transition")
    axis.legend()
    axis.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parents[1] / "evidence")
    args = parser.parse_args()
    metropolis = {side: read_window(args.artifacts / f"window-l{side}", side, "metropolis")
                  for side in (32, 64)}
    wolff = {side: read_window(args.artifacts / f"wolff-l{side}", side, "wolff")
             for side in (32, 64)}
    peaks, distance = plot_magnetization(metropolis, wolff, args.evidence / "magnetization-compare.png")
    ratio = plot_work(metropolis[64], wolff[64], args.evidence / "tau-compare.png")
    all_metropolis = {
        side: combine_runs(
            read_run(args.artifacts / f"coarse-l{side}", side),
            {t: np.abs(samples[0]) for t, samples in metropolis[side].items()},
        )
        for side in (32, 64)
    }
    plot_metropolis_tau(all_metropolis, args.evidence / "tau.png")
    print(f"Wolff peaks: L=32 {peaks[32]:.6f}, L=64 {peaks[64]:.6f}")
    print(f"Extrapolated Tc = {2 * peaks[64] - peaks[32]:.6f}; exact Tc = {EXACT_TC:.5f}")
    print(f"L=64 T=2.3 mean |M| standardized distance = {distance:.3f}")
    print(f"L=64 T=2.3 Metropolis/Wolff spin-update work ratio = {ratio:.1f}")
    for block_length in BLOCK_LENGTHS:
        peaks32, _, _ = bootstrap_fits(
            {t: wolff[32][t][0] for t in wolff[32]}, 32, block_length, 2026 + block_length
        )
        peaks64, _, _ = bootstrap_fits(
            {t: wolff[64][t][0] for t in wolff[64]}, 64, block_length, 3026 + block_length
        )
        count = min(len(peaks32), len(peaks64))
        tc_error = np.std(2.0 * peaks64[:count] - peaks32[:count], ddof=1)
        print(f"Wolff Tc bootstrap error, block {block_length}: {tc_error:.6f}")


if __name__ == "__main__":
    main()
