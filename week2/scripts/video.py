"""Render every saved fluid frame beside its cumulative radial distribution."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


def render(folder, output):
    folder = Path(folder)
    metadata = json.loads((folder / "run.json").read_text())
    frames = [json.loads(line) for line in (folder / "traj.jsonl").read_text().splitlines()]
    box = np.asarray(metadata["box"], dtype=float)
    n = metadata["n"]
    rmax = min(box) / 2
    bins = np.linspace(0, rmax, 51)
    radius = (bins[:-1] + bins[1:]) / 2
    shell_area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    counts = np.zeros(len(radius))
    pair_indices = np.triu_indices(n, 1)

    fig, (ax_atoms, ax_rdf) = plt.subplots(1, 2, figsize=(6.4, 3.6), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.88, wspace=0.32)
    ax_atoms.set_xlim(0, box[0])
    ax_atoms.set_ylim(0, box[1])
    ax_atoms.set_aspect("equal")
    ax_atoms.set_title("Atoms in periodic box")
    dots = ax_atoms.scatter([], [], s=12, color="#2067a2")
    ax_rdf.set_xlim(0, rmax)
    ax_rdf.set_ylim(0, 5)
    ax_rdf.set_xlabel("Pair distance r")
    ax_rdf.set_ylabel("g(r)")
    ax_rdf.set_title("Cumulative pair structure")
    ax_rdf.axhline(1, color="gray", lw=0.8, ls="--")
    line, = ax_rdf.plot(radius, np.zeros_like(radius), color="#c24b2d")

    writer = FFMpegWriter(fps=12, codec="libx264", bitrate=450)
    with writer.saving(fig, str(output), 100):
        for index, frame in enumerate(frames, 1):
            pos = np.asarray(frame["pos"], dtype=float)
            delta = pos[:, None, :] - pos[None, :, :]
            delta -= box * np.round(delta / box)
            distances = np.linalg.norm(delta, axis=2)[pair_indices]
            counts += 2 * np.histogram(distances, bins=bins)[0]
            rdf = counts / (index * n * metadata["rho"] * shell_area)
            dots.set_offsets(pos)
            line.set_ydata(rdf)
            ax_atoms.set_xlabel(f"t = {frame['t']:.2f}")
            writer.grab_frame()
    plt.close(fig)
    if Path(output).stat().st_size >= 5_000_000:
        raise ValueError("video exceeds the 5 MB evidence limit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("output")
    args = parser.parse_args()
    render(args.folder, args.output)


if __name__ == "__main__":
    main()
