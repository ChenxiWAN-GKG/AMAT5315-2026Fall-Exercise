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


def run_case(name: str, integrator: str, derivative: str, dt: float,
             sigma: float, nu: float, final_time: float) -> tuple[np.ndarray, np.ndarray, float]:
    command = [
        "cargo", "run", "--quiet", "--manifest-path", str(ROOT / "Cargo.toml"),
        "--bin", "line-accuracy-data", "--",
        "--n", "64", "--c", "1", "--nu", str(nu), "--sigma", str(sigma),
        "--final-time", str(final_time), "--dt", str(dt),
        "--integrator", integrator, "--derivative", derivative,
    ]
    text = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(text)))
    x = np.array([float(row["x"]) for row in rows])
    values = np.array([float(row["u"]) for row in rows])
    error = np.max(np.abs(values - exact_solution(x, sigma, nu, final_time)))
    print(f"{name}: maximum error = {error:.8e}")
    return x, values, error


def exact_solution(x: np.ndarray, sigma0: float, nu: float, time: float) -> np.ndarray:
    sigma = np.sqrt(sigma0**2 + 2.0 * nu * time)
    center = np.pi / 2.0 + time
    return sum(
        (sigma0 / sigma) * np.exp(-(x - center + 2.0 * np.pi * image) ** 2 / (2.0 * sigma**2))
        for image in range(-3, 4)
    )


def main() -> None:
    x, exact = None, None
    profile_cases = [
        ("RK4 Fourier", "rk4", "fourier", 0.02),
        ("RK4 centred differences", "rk4", "centered", 0.02),
        ("Euler Fourier", "euler", "fourier", 0.005),
    ]
    results = []
    for name, integrator, derivative, dt in profile_cases:
        x, values, _ = run_case(name, integrator, derivative, dt, 0.25, 0.002, 2.0 * np.pi)
        if exact is None:
            exact = exact_solution(x, 0.25, 0.002, 2.0 * np.pi)
        results.append((name, values))
    convergence_cases = [
        ("Euler", "euler"),
        ("midpoint", "midpoint"),
        ("RK4", "rk4"),
        ("equal-weight RK4", "equal-rk4"),
    ]
    steps = np.array([0.02, 0.01, 0.005, 0.0025])
    convergence = {}
    for name, integrator in convergence_cases:
        errors = []
        for dt in steps:
            _, _, error = run_case(f"{name}, dt={dt:g}", integrator, "fourier", dt, 0.35, 0.05, 1.0)
            errors.append(error)
        convergence[name] = np.array(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    axes[0].plot(x, exact, "k--", linewidth=2, label="exact")
    for name, values in results:
        axes[0].plot(x, values, linewidth=1.5, label=name)
    axes[0].set(xlabel="x", ylabel="u(x, 2π)", title="One lap of a periodic Gaussian pulse")
    axes[0].legend()
    axes[0].set_xlim(0.0, 2.0 * np.pi)

    for name, errors in convergence.items():
        slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
        fit = np.exp(intercept) * steps ** slope
        axes[1].loglog(steps, errors, "o", label=f"{name}, slope {slope:.2f}")
        axes[1].loglog(steps, fit, "-")
        print(f"{name}: fitted slope = {slope:.6f}")
    axes[1].set(xlabel="time step h", ylabel="maximum error at t = 1",
                title="Fourier pulse accuracy")
    axes[1].legend(fontsize=8)
    fig.savefig(OUTPUT, dpi=180)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
