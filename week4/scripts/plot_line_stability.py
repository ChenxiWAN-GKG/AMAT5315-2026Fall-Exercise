"""Create the Week 4 stability map and pulse-above/below-limit figure."""

from pathlib import Path
import csv
import io
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT = ROOT / "evidence" / "line-stability.png"


def run_binary(name: str, arguments: list[str]) -> str:
    command = [
        "cargo", "run", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"),
        "--bin", name, "--", *arguments,
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def stability_grid() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    text = run_binary("stability-data", [
        "--xmin", "-4.5", "--xmax", "1.5", "--ymin", "-4.0", "--ymax", "4.0", "--points", "401",
    ])
    rows = list(csv.DictReader(io.StringIO(text)))
    points = int(np.sqrt(len(rows)))
    real = np.array([float(row["re"]) for row in rows]).reshape(points, points)
    imaginary = np.array([float(row["im"]) for row in rows]).reshape(points, points)
    growth = {name: np.array([float(row[name]) for row in rows]).reshape(points, points)
              for name in ("euler", "midpoint", "rk4")}
    return real, imaginary, growth


def pulse_data(dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = run_binary("line-data", [
        "--n", "64", "--c", "1", "--nu", "0.05", "--dt", str(dt),
        "--final-time", "6", "--sigma", "0.35",
    ])
    rows = list(csv.DictReader(io.StringIO(text)))
    times = np.array(sorted({float(row["t"]) for row in rows}))
    x = np.array([float(row["x"]) for row in rows[:64]])
    values = np.array([float(row["u"]) for row in rows[64:]]).reshape(len(times) - 1, 64)
    return times[1:], x, values


def main() -> None:
    real, imaginary, growth = stability_grid()
    n = 64
    nu = 0.05
    c = 1.0
    modes = np.arange(-n // 2, n // 2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    image = axes[0].pcolormesh(
        real, imaginary, growth["rk4"], shading="auto", cmap="viridis",
        norm=LogNorm(vmin=0.1, vmax=10), rasterized=True,
    )
    levels = [1.0]
    for name, style in (("euler", "--"), ("midpoint", "-.") , ("rk4", "-")):
        axes[0].contour(real, imaginary, growth[name], levels=levels, colors="black", linestyles=style)
    for dt, color in ((0.045, "white"), (0.056, "red")):
        real_modes = -nu * modes.astype(float) ** 2 * dt
        imaginary_modes = -c * modes.astype(float) * dt
        nyquist = modes == -n // 2
        imaginary_modes[nyquist] = 0.0
        axes[0].scatter(real_modes, imaginary_modes, s=12, color=color, edgecolors="black", linewidths=0.3,
                        label=f"line modes, h={dt}")
    axes[0].set(xlabel="Re(z)", ylabel="Im(z)", title="Measured stability growth")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.colorbar(image, ax=axes[0], label="RK4 growth per step")

    for axis, dt in zip(axes[1:], (0.045, 0.056)):
        times, x, values = pulse_data(dt)
        axis.imshow(values, extent=(x[0], 2 * np.pi, times[-1], times[0]), aspect="auto", cmap="magma")
        axis.set(xlabel="x", ylabel="t", title=f"RK4 pulse, h={dt}")
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
