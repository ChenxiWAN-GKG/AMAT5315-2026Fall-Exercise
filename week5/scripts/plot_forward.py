"""Plot receiver gathers from a forward seismic run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_gathers(experiment_path: str | Path, traces_path: str | Path, output_path: str | Path) -> None:
    """Save one receiver gather panel for each shot."""
    experiment = json.loads(Path(experiment_path).read_text())
    traces = np.load(traces_path)
    receivers = np.asarray(experiment["receivers"], dtype=float)
    receiver_x_km = receivers[:, 0] * experiment["length_unit_m"] / 1000.0
    time_seconds = (
        np.arange(experiment["steps"]) * experiment["dt"] * experiment["time_unit_s"]
    )

    figure, axes = plt.subplots(
        traces.shape[0], 1, figsize=(8, 9), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    maximum = np.max(np.abs(traces))
    for shot_index, axis in enumerate(axes):
        image = axis.imshow(
            traces[shot_index],
            aspect="auto",
            origin="upper",
            extent=[receiver_x_km[0], receiver_x_km[-1], time_seconds[-1], time_seconds[0]],
            cmap="seismic",
            vmin=-maximum,
            vmax=maximum,
        )
        axis.set_ylabel("Time (s)")
        axis.set_title(f"Shot {shot_index}; source x = {experiment['shots'][shot_index][0] * experiment['length_unit_m'] / 1000:.1f} km")
        figure.colorbar(image, ax=axis, label="Pressure")
    axes[-1].set_xlabel("Receiver position (km)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def print_summary(traces_path: str | Path) -> None:
    traces = np.load(traces_path)
    print(f"all-trace L2 norm: {np.linalg.norm(traces):.9f}")
    for shot_index, shot in enumerate(traces):
        time_index, receiver_index = np.unravel_index(np.argmax(np.abs(shot)), shot.shape)
        print(
            f"shot {shot_index}: max={shot[time_index, receiver_index]:.9f}, "
            f"step={time_index}, receiver={receiver_index}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot_gathers(args.experiment, args.traces, args.out)
    print_summary(args.traces)


if __name__ == "__main__":
    main()
