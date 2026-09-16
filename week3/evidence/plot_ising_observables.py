#!/usr/bin/env python3
"""Plot Ising magnetization and susceptibility from recorded Metropolis ramps."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TC = 2.0 / math.log1p(math.sqrt(2.0))


def read_observables(path: Path) -> dict[float, dict[str, float]]:
    with (path / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    side = int(metadata["L"])
    expected_measure = int(metadata["measure"])
    by_temperature: dict[float, list[float]] = {}
    with (path / "series.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            temperature = float(record["T"])
            by_temperature.setdefault(temperature, []).append(float(record["M"]))

    observables = {}
    for temperature, values in sorted(by_temperature.items()):
        if len(values) != expected_measure:
            raise ValueError(
                f"{path}: T={temperature:g} has {len(values)} rows, "
                f"expected {expected_measure}"
            )
        mean_abs_m = fmean(abs(value) for value in values)
        mean_m2 = fmean(value * value for value in values)
        observables[temperature] = {
            "mean_abs_m": mean_abs_m,
            "susceptibility": side * side * (mean_m2 - mean_abs_m * mean_abs_m) / temperature,
        }
    if not observables:
        raise ValueError(f"{path}: no measured rows found")
    return observables


def onsager_magnetization(temperatures):
    values = []
    for temperature in temperatures:
        if temperature >= TC:
            values.append(0.0)
        else:
            values.append((1.0 - math.sinh(2.0 / temperature) ** -4) ** 0.125)
    return values


def plot_magnetization(
    coarse_l64: dict[float, dict[str, float]],
    window_l64: dict[float, dict[str, float]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for values, label, style in (
        (coarse_l64, "Metropolis L=64, coarse", "o-"),
        (window_l64, "Metropolis L=64, transition window", "s--"),
    ):
        temperatures = list(values)
        axis.plot(
            temperatures,
            [values[t]["mean_abs_m"] for t in temperatures],
            style,
            linewidth=1.4,
            markersize=3.5,
            label=label,
        )

    reference_temperatures = [1.5 + 2.0 * index / 600.0 for index in range(601)]
    axis.plot(
        reference_temperatures,
        onsager_magnetization(reference_temperatures),
        color="black",
        linewidth=1.8,
        label="Onsager infinite lattice",
    )
    axis.axvline(TC, color="0.45", linestyle=":", linewidth=1.2, label=fr"$T_c={TC:.4f}$")
    axis.set_xlabel("Temperature T")
    axis.set_ylabel(r"mean $|M|$")
    axis.set_title("2D Ising magnetization: L = 64")
    axis.set_xlim(1.5, 3.5)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_susceptibility(
    datasets: list[tuple[str, dict[float, dict[str, float]], str, str]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for label, values, color, style in datasets:
        temperatures = list(values)
        axis.plot(
            temperatures,
            [values[t]["susceptibility"] for t in temperatures],
            style,
            color=color,
            linewidth=1.4,
            markersize=3.2,
            label=label,
        )
    axis.axvline(TC, color="0.45", linestyle=":", linewidth=1.2, label=fr"$T_c={TC:.4f}$")
    axis.set_xlabel("Temperature T")
    axis.set_ylabel(r"$\chi(T)=L^2[\langle M^2\rangle-\langle |M|\rangle^2]/T$")
    axis.set_title("2D Ising susceptibility")
    axis.set_xlim(1.5, 3.5)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
        help="directory containing coarse-l32, coarse-l64, window-l32, and window-l64",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for magnetization.png and susceptibility.png",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coarse_l32 = read_observables(args.artifacts / "coarse-l32")
    coarse_l64 = read_observables(args.artifacts / "coarse-l64")
    window_l32 = read_observables(args.artifacts / "window-l32")
    window_l64 = read_observables(args.artifacts / "window-l64")

    plot_magnetization(coarse_l64, window_l64, args.out_dir / "magnetization.png")
    plot_susceptibility(
        [
            ("L=32, coarse", coarse_l32, "tab:orange", "o-"),
            ("L=32, transition window", window_l32, "tab:orange", "s--"),
            ("L=64, coarse", coarse_l64, "tab:blue", "o-"),
            ("L=64, transition window", window_l64, "tab:blue", "s--"),
        ],
        args.out_dir / "susceptibility.png",
    )
    print(f"Tc = {TC:.12f}")
    print(f"wrote {args.out_dir / 'magnetization.png'}")
    print(f"wrote {args.out_dir / 'susceptibility.png'}")


if __name__ == "__main__":
    main()
