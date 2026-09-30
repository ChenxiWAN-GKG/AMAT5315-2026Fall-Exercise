"""Plot the reflector experiment geometry and its source pulse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ricker(time: np.ndarray, frequency: float, peak_time: float) -> np.ndarray:
    theta = np.pi * frequency * (time - peak_time)
    return (1.0 - 2.0 * theta**2) * np.exp(-theta**2)


def plot_inputs(experiment_path: str | Path, output_path: str | Path) -> None:
    """Save the experiment geometry and source-pulse figure."""
    experiment = json.loads(Path(experiment_path).read_text())
    nx = experiment["nx"]
    nz = experiment["nz"]
    length_scale = experiment["length_unit_m"] / 1000.0
    time_scale = experiment["time_unit_s"]
    x_extent = nx * experiment["dx"] * length_scale
    z_extent = nz * experiment["dx"] * length_scale
    background = np.asarray(experiment["background"], dtype=float)
    perturbation = np.asarray(experiment["perturbation"], dtype=float)

    figure, (geometry_axis, pulse_axis) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )
    image = geometry_axis.imshow(
        background,
        origin="upper",
        extent=[0.0, x_extent, z_extent, 0.0],
        aspect="equal",
        cmap="viridis",
    )
    figure.colorbar(image, ax=geometry_axis, label="Background speed (km/s)")
    nonzero = np.abs(perturbation[np.nonzero(perturbation)])
    if nonzero.size:
        level = float(np.max(nonzero) * 0.5)
        geometry_axis.contour(
            perturbation,
            levels=[-level, level],
            origin="upper",
            extent=[0.0, x_extent, z_extent, 0.0],
            colors=["royalblue", "crimson"],
            linewidths=1.5,
        )
    shots = np.asarray(experiment["shots"], dtype=float)
    receivers = np.asarray(experiment["receivers"], dtype=float)
    geometry_axis.scatter(
        shots[:, 0] * experiment["dx"] * length_scale,
        shots[:, 1] * experiment["dx"] * length_scale,
        marker="*",
        s=100,
        color="black",
        label="Shots",
        zorder=3,
    )
    geometry_axis.scatter(
        receivers[:, 0] * experiment["dx"] * length_scale,
        receivers[:, 1] * experiment["dx"] * length_scale,
        marker="v",
        s=30,
        color="white",
        edgecolor="black",
        label="Receivers",
        zorder=3,
    )
    geometry_axis.set_xlabel("Horizontal position (km)")
    geometry_axis.set_ylabel("Depth (km)")
    geometry_axis.set_title("Reflector experiment geometry")
    geometry_axis.legend(loc="lower right")

    reduced_time = np.arange(experiment["steps"], dtype=float) * experiment["dt"]
    pulse = experiment["source_amplitude"] * ricker(
        reduced_time, experiment["source_frequency"], experiment["source_peak_time"]
    )
    pulse_axis.plot(reduced_time * time_scale, pulse, color="darkorange")
    pulse_axis.axhline(0.0, color="black", linewidth=0.7)
    pulse_axis.axvline(
        experiment["source_peak_time"] * time_scale,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Peak time",
    )
    pulse_axis.set_xlabel("Time (s)")
    pulse_axis.set_ylabel("Source amplitude")
    pulse_axis.set_title("Ricker source pulse")
    pulse_axis.legend()
    pulse_axis.grid(alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="inputs/reflector.json")
    parser.add_argument("--out", default="artifacts/inputs.png")
    args = parser.parse_args()
    plot_inputs(args.experiment, args.out)


if __name__ == "__main__":
    main()
