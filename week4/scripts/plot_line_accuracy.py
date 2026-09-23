"""Compare three library runs with the exact periodic Gaussian solution."""

from pathlib import Path
import csv
import io
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT = ROOT / "evidence" / "line-accuracy.png"


def run_case(name: str, integrator: str, derivative: str, dt: float) -> tuple[np.ndarray, np.ndarray]:
    command = [
        "cargo", "run", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"),
        "--bin", "line-accuracy-data", "--",
        "--n", "64", "--c", "1", "--nu", "0.002", "--sigma", "0.25",
        "--final-time", str(2.0 * np.pi), "--dt", str(dt),
        "--integrator", integrator, "--derivative", derivative,
    ]
    text = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(text)))
    x = np.array([float(row["x"]) for row in rows])
    values = np.array([float(row["u"]) for row in rows])
    error = np.max(np.abs(values - exact_solution(x)))
    print(f"{name}: maximum error = {error:.8e}")
    return x, values


def exact_solution(x: np.ndarray) -> np.ndarray:
    sigma0 = 0.25
    nu = 0.002
    time = 2.0 * np.pi
    sigma = np.sqrt(sigma0**2 + 2.0 * nu * time)
    center = np.pi / 2.0 + time
    return sum(
        (sigma0 / sigma) * np.exp(-(x - center + 2.0 * np.pi * image) ** 2 / (2.0 * sigma**2))
        for image in range(-3, 4)
    )


def main() -> None:
    x, exact = None, None
    cases = [
        ("RK4 Fourier", "rk4", "fourier", 0.02),
        ("RK4 centred differences", "rk4", "centered", 0.02),
        ("Euler Fourier", "euler", "fourier", 0.005),
    ]
    results = []
    for name, integrator, derivative, dt in cases:
        x, values = run_case(name, integrator, derivative, dt)
        if exact is None:
            exact = exact_solution(x)
        results.append((name, values))
    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.plot(x, exact, "k--", linewidth=2, label="exact")
    for name, values in results:
        axis.plot(x, values, linewidth=1.5, label=name)
    axis.set(xlabel="x", ylabel="u(x, 2π)", title="One lap of a periodic Gaussian pulse")
    axis.legend()
    axis.set_xlim(0.0, 2.0 * np.pi)
    fig.savefig(OUTPUT, dpi=180)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
