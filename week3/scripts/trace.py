#!/usr/bin/env python3
"""Plot the first 2000 recorded |M| sweeps at two temperatures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SWEEPS_TO_PLOT = 2000


def read_temperature(path: Path, temperature: float, expected_side: int) -> list[dict]:
    with (path / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["update"] != "metropolis":
        raise ValueError(f"{path}: expected a Metropolis run")
    if int(metadata["L"]) != expected_side:
        raise ValueError(f"{path}: expected L={expected_side}")

    records = []
    with (path / "series.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if float(record["T"]) == temperature:
                records.append(record)
    if len(records) < SWEEPS_TO_PLOT:
        raise ValueError(
            f"{path}: T={temperature:g} has only {len(records)} recorded rows, "
            f"expected at least {SWEEPS_TO_PLOT}"
        )
    return records


def first_sweeps(records: list[dict], count: int) -> tuple[np.ndarray, np.ndarray]:
    selected = records[:count]
    sweeps = np.asarray([int(record["sweep"]) for record in selected])
    magnetization = np.asarray([abs(float(record["M"])) for record in selected])
    return sweeps, magnetization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "evidence" / "trace.png"
    )
    args = parser.parse_args()

    cases = (
        (2.3, args.artifacts / "window-l64"),
        (3.0, args.artifacts / "coarse-l64"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, constrained_layout=True)
    for axis, (temperature, path) in zip(axes, cases):
        records = read_temperature(path, temperature, 64)
        sweeps, magnetization = first_sweeps(records, SWEEPS_TO_PLOT)
        axis.plot(sweeps, magnetization, linewidth=0.7, color="tab:blue")
        axis.axhline(float(np.mean(magnetization)), color="tab:orange", linewidth=1.2, linestyle="--")
        axis.set_ylabel(r"$|M|$")
        axis.set_title(fr"Metropolis, $L=64$, $T={temperature:.1f}$")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
        axis.text(
            0.99,
            0.04,
            fr"mean $|M|$ = {np.mean(magnetization):.4f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
        )
    axes[-1].set_xlabel("Recorded sweep")
    figure.savefig(args.out, dpi=180)
    plt.close(figure)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
