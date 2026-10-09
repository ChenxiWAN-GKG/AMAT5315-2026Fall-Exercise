"""Recompute fluid diagnostics directly from the saved positions and velocities."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def diagnostics(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "run.json").read_text())
    frames = [json.loads(line) for line in (folder / "traj.jsonl").read_text().splitlines()]
    if not frames:
        raise ValueError("trajectory has no saved frames")
    box = np.asarray(metadata["box"])
    cutoff_energy = 4 * (2.5 ** -12 - 2.5 ** -6)
    totals = []
    speeds2 = []
    stored_error = 0.0
    for frame in frames:
        pos = np.asarray(frame["pos"], dtype=float)
        vel = np.asarray(frame["vel"], dtype=float)
        if pos.shape != (metadata["n"], 2) or vel.shape != pos.shape:
            raise ValueError("frame particle count does not match run metadata")
        delta = pos[:, None, :] - pos[None, :, :]
        delta -= box * np.round(delta / box)
        r2 = np.sum(delta * delta, axis=2)
        pair_r2 = r2[np.triu_indices(metadata["n"], 1)]
        if np.any(pair_r2 == 0):
            raise ValueError("overlapping atoms")
        inside = pair_r2 < 2.5**2
        inverse_six = pair_r2[inside] ** -3
        e_pot = float(np.sum(4 * (inverse_six**2 - inverse_six) - cutoff_energy))
        v2 = np.sum(vel * vel, axis=1)
        e_kin = float(np.sum(v2) / 2)
        totals.append(e_pot + e_kin)
        speeds2.extend(v2)
        stored_error = max(stored_error, abs(e_pot - frame["E_pot"]), abs(e_kin - frame["E_kin"]))

    k = max(1, len(totals) // 10)
    drift = abs(float(np.mean(totals[-k:]) - np.mean(totals[:k]))) / abs(totals[0])
    t_speed = float(np.mean(speeds2) / 2)
    edges = np.sqrt(-2 * t_speed * np.log1p(-np.arange(1, 24) / 24))
    observed = np.histogram(np.sqrt(speeds2), bins=np.r_[0, edges, np.inf])[0]
    expected = len(speeds2) / 24
    chi2_22 = float(np.sum((observed - expected) ** 2 / expected) / 22)
    return drift, t_speed, chi2_22, stored_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    drift, temperature, chi2, stored = diagnostics(args.folder)
    for name, value, limit in [
        ("secular_drift", drift, 2e-3),
        ("abs(T_speed - 0.5)", abs(temperature - 0.5), 0.05),
        ("chi2/22", chi2, 2.0),
    ]:
        print(f"{name} = {value:.8g}; limit < {limit:g}; {'PASS' if value < limit else 'FAIL'}")
    print(f"T_speed = {temperature:.8g}; max stored-energy difference = {stored:.3g}")


if __name__ == "__main__":
    main()
