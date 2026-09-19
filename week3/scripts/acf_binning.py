#!/usr/bin/env python3
"""Plot the |M| autocorrelation and binning error at the critical window."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TEMPERATURE = 2.3
SIDE = 64
BLOCK_LENGTHS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def read_magnetization(path: Path) -> np.ndarray:
    with (path / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata["update"] != "metropolis" or int(metadata["L"]) != SIDE:
        raise ValueError(f"{path}: expected a Metropolis L={SIDE} run")

    values = []
    with (path / "series.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if float(record["T"]) == TEMPERATURE:
                values.append(abs(float(record["M"])))
    if not values:
        raise ValueError(f"{path}: no rows at T={TEMPERATURE:g}")
    return np.asarray(values, dtype=np.float64)


def autocorrelation(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    variance = np.mean(centered * centered)
    if variance == 0.0:
        return np.ones(len(values))

    fft_length = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    covariance = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_length)[: len(values)]
    covariance /= np.arange(len(values), 0, -1)
    return covariance / covariance[0]


def block_error(values: np.ndarray, block_length: int) -> tuple[float, int]:
    block_count = len(values) // block_length
    if block_count < 2:
        raise ValueError(f"not enough complete blocks of length {block_length}")
    usable = values[: block_count * block_length]
    means = usable.reshape(block_count, block_length).mean(axis=1)
    error = np.std(means, ddof=1) / np.sqrt(block_count)
    return float(error), block_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "evidence" / "acf-binning.png"
    )
    args = parser.parse_args()

    values = read_magnetization(args.artifacts / "window-l64")
    acf = autocorrelation(values)
    block_results = [block_error(values, length) for length in BLOCK_LENGTHS]

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    lag_limit = min(3000, len(acf) - 1)
    axes[0].plot(np.arange(lag_limit + 1), acf[: lag_limit + 1], color="tab:blue", linewidth=0.8)
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_xlabel("Lag (sweeps)")
    axes[0].set_ylabel(r"autocorrelation $\rho(t)$")
    axes[0].set_title(r"Metropolis $|M|$ autocorrelation: $L=64$, $T=2.3$")
    axes[0].grid(True, alpha=0.25)

    lengths = np.asarray(BLOCK_LENGTHS)
    errors = np.asarray([error for error, _ in block_results])
    block_counts = [count for _, count in block_results]
    axes[1].plot(lengths, errors, "o-", color="tab:orange", linewidth=1.2, markersize=4)
    for length, error, count in zip(lengths, errors, block_counts):
        axes[1].annotate(
            f"{count} blocks",
            (length, error),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Block length (sweeps, logarithmic scale)")
    axes[1].set_ylabel(r"standard error of mean $|M|$")
    axes[1].set_title("Binning estimate of the mean error")
    axes[1].grid(True, alpha=0.25, which="both")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)

    print(f"wrote {args.out}")
    for length, error, count in zip(BLOCK_LENGTHS, errors, block_counts):
        print(f"block_length={length} blocks={count} standard_error={error:.8f}")
    if np.all(np.diff(errors) > 0.0):
        print("sampling error unresolved: the binning curve is still rising")


if __name__ == "__main__":
    main()
