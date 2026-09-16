#!/usr/bin/env python3
"""Estimate statistical errors for |M| in the recorded Metropolis runs."""

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np


BLOCK_COUNT = 50


def read_run(path: Path, expected_side: int) -> dict[float, np.ndarray]:
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
            temperature = float(record["T"])
            values.setdefault(temperature, []).append(abs(float(record["M"])))

    expected_measure = int(metadata["measure"])
    result = {}
    for temperature, samples in sorted(values.items()):
        if len(samples) != expected_measure:
            raise ValueError(
                f"{path}: T={temperature:g} has {len(samples)} rows, "
                f"expected {expected_measure}"
            )
        result[temperature] = np.asarray(samples, dtype=np.float64)
    return result


def autocorrelation_time(samples: np.ndarray) -> float:
    count = len(samples)
    centered = samples - fmean(samples)
    variance = float(np.mean(centered * centered))
    if variance == 0.0:
        return 0.5

    fft_length = 1 << (2 * count - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    autocovariance = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_length)[:count]
    autocovariance /= np.arange(count, 0, -1)
    autocorrelation = autocovariance / autocovariance[0]

    tau = 0.5
    for lag in range(1, count):
        if lag > 6.0 * tau:
            break
        tau += float(autocorrelation[lag])
    return max(0.5, tau)


def error_row(samples: np.ndarray) -> tuple[float, float, float, float]:
    count = len(samples)
    naive = float(np.std(samples, ddof=1) / np.sqrt(count))
    if count % BLOCK_COUNT != 0:
        raise ValueError(f"{count} samples cannot form {BLOCK_COUNT} equal blocks")
    block_size = count // BLOCK_COUNT
    block_means = samples.reshape(BLOCK_COUNT, block_size).mean(axis=1)
    block_error = float(np.std(block_means, ddof=1) / np.sqrt(BLOCK_COUNT))
    ratio = block_error / naive if naive != 0.0 else float("nan")
    return naive, block_error, ratio, autocorrelation_time(samples)


def combine_runs(coarse: dict[float, np.ndarray], window: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    combined = dict(coarse)
    combined.update(window)
    return dict(sorted(combined.items()))


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
        default=Path(__file__).resolve().parents[1] / "evidence" / "errors.txt",
    )
    args = parser.parse_args()

    runs = {
        32: combine_runs(
            read_run(args.artifacts / "coarse-l32", 32),
            read_run(args.artifacts / "window-l32", 32),
        ),
        64: combine_runs(
            read_run(args.artifacts / "coarse-l64", 64),
            read_run(args.artifacts / "window-l64", 64),
        ),
    }
    lines = [
        "# Observable: |M|; window runs replace coarse runs at overlapping temperatures",
        "# Block standard error: 50 contiguous equal-size block averages",
        "# tau_int: 0.5 + sum rho(lag), stopping when lag > 6*tau_int",
        "L\tT\tmean_abs_M\tnaive_se\tblock50_se\tblock50_over_naive\ttau_int",
    ]
    for side in (32, 64):
        for temperature, samples in runs[side].items():
            naive, block_error, ratio, tau = error_row(samples)
            lines.append(
                f"{side}\t{temperature:.6f}\t{fmean(samples):.8f}\t"
                f"{naive:.8f}\t{block_error:.8f}\t{ratio:.8f}\t{tau:.8f}"
            )

    output = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
