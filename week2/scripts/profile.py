"""Summarize saved samply call stacks by inclusive force-function samples."""

import gzip
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(profile, binary):
    with gzip.open(profile, "rt") as source:
        thread = json.load(source)["threads"][0]
    strings = thread["stringArray"]
    frames = thread["frameTable"]
    functions = thread["funcTable"]
    indices = []
    addresses = []
    for index, function_index in enumerate(frames["func"]):
        name = strings[functions["name"][function_index]]
        if name.startswith("0x"):
            indices.append(index)
            addresses.append(hex(0x100000000 + int(name, 16)))
    names = subprocess.check_output(
        ["atos", "-o", str(binary), "-arch", "arm64", "-l", "0x100000000", *addresses],
        text=True,
    ).splitlines()
    symbol = dict(zip(indices, names))
    stack = thread["stackTable"]
    force_samples = 0
    for entry in thread["samples"]["stack"]:
        while entry is not None:
            frame = stack["frame"][entry]
            if "forces_and_energy_with_method" in symbol.get(frame, ""):
                force_samples += 1
                break
            entry = stack["prefix"][entry]
    total = len(thread["samples"]["stack"])
    duration = (thread["samples"]["time"][-1] - thread["samples"]["time"][0]) / 1000
    return force_samples, total, duration


def main():
    week = Path(__file__).resolve().parents[1]
    binary = week / "md/target/release/md"
    for method in ["naive", "cells"]:
        count, total, duration = summarize(week / f"profile-{method}.json.gz", binary)
        share = 100 * count / total
        fig, ax = plt.subplots(figsize=(6.0, 2.5))
        ax.barh([0.45], [share], height=0.35, color="#246da5")
        ax.barh([0.45], [100 - share], left=[share], height=0.35, color="#ce8654")
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.2, 0.8)
        ax.set_yticks([])
        ax.set_xlabel("Share of sampled call stacks (%)")
        ax.set_title(f"Samply: {method} force search, N = 400")
        ax.text(share / 2, 0.45, f"Force function and calls: {share:.1f}%", ha="center", va="center", color="white", weight="bold")
        ax.text(50, 0.05, f"{count}/{total} samples · profiled run {duration:.3f} s", ha="center")
        fig.tight_layout()
        fig.savefig(week / f"profile-{method}.png", dpi=180)
        plt.close(fig)
        print(f"{method}: force share {share:.1f}% ({count}/{total}); profiled run {duration:.3f} s")


if __name__ == "__main__":
    main()
