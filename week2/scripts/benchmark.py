"""Measure the Week 2 timing and force-scaling tables."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN = [
    "--rho", "0.8", "--temperature", "0.5", "--dt", "0.01",
    "--sample-every", "50", "--seed", "2026",
]


def measured(command, workdir):
    start = time.perf_counter()
    subprocess.run(command, cwd=workdir, stdout=subprocess.DEVNULL, check=True)
    return time.perf_counter() - start


def repeat(command, workdir):
    return [measured(command, workdir) for _ in range(3)]


def summary(values):
    return f"{statistics.median(values):.3f} ({min(values):.3f}–{max(values):.3f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="week2-sim.py")
    args = parser.parse_args()
    week = Path(__file__).resolve().parents[1]
    baseline = Path(args.baseline).resolve()
    debug = week / "md/target/debug/md"
    release = week / "md/target/release/md"
    with tempfile.TemporaryDirectory(prefix="amat5315-md-benchmark-") as temp:
        temp = Path(temp)
        full = ["--n", "100", *RUN, "--eq-steps", "2000", "--steps", "10000", "--out", str(temp / "run")]
        timing = {
            "NumPy": repeat([sys.executable, str(baseline)], temp),
            "Rust debug": repeat([str(debug), *full], temp),
            "Rust release": repeat([str(release), *full], temp),
        }
        scaling = {}
        for n in [100, 400, 1600]:
            scaling[str(n)] = {}
            for force in ["naive", "cells"]:
                command = [str(release), "--n", str(n), *RUN, "--eq-steps", "100", "--steps", "500", "--force", force, "--out", str(temp / "run")]
                scaling[str(n)][force] = repeat(command, temp)

    data = {"timing_seconds": timing, "scaling_seconds": scaling}
    (week / "benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Timing: median (min–max) seconds")
    for label, values in timing.items():
        print(f"{label}: {summary(values)}")
    print("Scaling: median (min–max) seconds for 600 steps")
    for n, runs in scaling.items():
        a, b = runs["naive"], runs["cells"]
        speedup = statistics.median(a) / statistics.median(b)
        print(f"N={n}: naive {summary(a)}; cells {summary(b)}; speedup {speedup:.2f}×")

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for method in ["naive", "cells"]:
        n_values = [100, 400, 1600]
        seconds_per_step = [statistics.median(scaling[str(n)][method]) / 600 for n in n_values]
        ax.loglog(n_values, seconds_per_step, "o-", label=method)
    ax.set_xticks([100, 400, 1600], ["100", "400", "1600"])
    ax.set_xlabel("Atoms N")
    ax.set_ylabel("Seconds per step")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(week / "scaling.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
