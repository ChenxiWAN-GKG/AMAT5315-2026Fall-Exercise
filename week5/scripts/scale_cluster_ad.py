"""Benchmark forward- and reverse-mode AD for Lennard-Jones clusters."""

from __future__ import annotations

import math
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_cluster(n_atoms: int, seed: int = 0) -> jnp.ndarray:
    """Make a perturbed cubic lattice with exactly n_atoms points."""
    spacing = 2 ** (1 / 6)
    side = math.ceil(n_atoms ** (1 / 3))
    axis = jnp.arange(side, dtype=jnp.float64)
    grid = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    coordinates = grid.reshape(-1, 3)[:n_atoms] * spacing
    key = jax.random.key(seed)
    displacement = 0.05 * jax.random.normal(key, coordinates.shape, dtype=jnp.float64)
    return coordinates + displacement


def cluster_energy(coordinates: jnp.ndarray) -> jnp.ndarray:
    """Compute the sum of pair energies over the upper triangle."""
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    upper = jnp.triu(jnp.ones((coordinates.shape[0], coordinates.shape[0]), dtype=bool), k=1)
    squared_distances = jnp.sum(differences**2, axis=-1)
    safe_distances = jnp.sqrt(jnp.where(upper, squared_distances, 1.0))
    pair_energy = 4 * (safe_distances**-12 - safe_distances**-6)
    return jnp.sum(jnp.where(upper, pair_energy, 0.0))


def analytic_gradient(coordinates: jnp.ndarray) -> jnp.ndarray:
    """Return the analytic gradient, the negative of the analytic forces."""
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    off_diagonal = ~jnp.eye(coordinates.shape[0], dtype=bool)
    squared_distances = jnp.sum(differences**2, axis=-1)
    safe_distances = jnp.sqrt(jnp.where(off_diagonal, squared_distances, 1.0))
    derivative = 24 * (safe_distances**-7 - 2 * safe_distances**-13)
    coefficient = jnp.where(off_diagonal, derivative / safe_distances, 0.0)
    return jnp.sum(coefficient[..., None] * differences, axis=1)


def _time_repeated(function, *arguments, repeats: int = 5) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        result = function(*arguments)
        jax.block_until_ready(result)
    return (time.perf_counter() - start) / repeats


def benchmark_size(n_atoms: int) -> dict[str, float | int]:
    """Benchmark both modes and compare their gradients with analytic forces."""
    coordinates = make_cluster(n_atoms)
    inputs = coordinates.size
    energy = jax.jit(cluster_energy)
    reverse = jax.jit(jax.grad(cluster_energy))

    def jvp(coordinates, direction):
        return jax.jvp(cluster_energy, (coordinates,), (direction,))[1]

    forward = jax.jit(jvp)
    direction = jnp.zeros_like(coordinates).at[0, 0].set(1.0)
    energy(coordinates).block_until_ready()
    reverse(coordinates).block_until_ready()
    forward(coordinates, direction).block_until_ready()

    energy_time = _time_repeated(energy, coordinates)
    reverse_time = _time_repeated(reverse, coordinates)

    directions = jnp.eye(inputs, dtype=jnp.float64).reshape(inputs, *coordinates.shape)
    start = time.perf_counter()
    forward_components = []
    for direction in directions:
        forward_components.append(forward(coordinates, direction))
        forward_components[-1].block_until_ready()
    forward_time = time.perf_counter() - start
    forward_gradient = jnp.stack(forward_components)

    reverse_gradient = reverse(coordinates)
    exact_gradient = analytic_gradient(coordinates)
    largest_force = jnp.max(jnp.abs(-exact_gradient))
    forward_error = jnp.max(jnp.abs(-forward_gradient.reshape(coordinates.shape) + exact_gradient))
    reverse_error = jnp.max(jnp.abs(-reverse_gradient + exact_gradient))

    return {
        "atoms": n_atoms,
        "inputs": inputs,
        "energy_time": energy_time,
        "forward_ratio": forward_time / energy_time,
        "reverse_ratio": reverse_time / energy_time,
        "forward_relative_error": float(forward_error / largest_force),
        "reverse_relative_error": float(reverse_error / largest_force),
    }


def run_scaling(
    output_path: str | Path, sizes: tuple[int, ...] = (64, 128, 256, 512, 1024)
) -> list[dict[str, float | int]]:
    """Run all requested sizes, save a log-log scaling plot, and return records."""
    records = [benchmark_size(n_atoms) for n_atoms in sizes]
    inputs = [record["inputs"] for record in records]
    forward_ratios = [record["forward_ratio"] for record in records]
    reverse_ratios = [record["reverse_ratio"] for record in records]

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.loglog(inputs, forward_ratios, "o-", label="forward mode")
    axis.loglog(inputs, reverse_ratios, "s-", label="reverse mode")
    axis.set_xlabel("Number of inputs (3N)")
    axis.set_ylabel("Gradient time / energy time")
    axis.set_title("Lennard-Jones cluster gradient scaling")
    axis.grid(alpha=0.3, which="both")
    axis.legend()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return records


def main() -> None:
    records = run_scaling("artifacts/ad/scaling.png")
    for record in records:
        print(
            f"N={record['atoms']:4d}, inputs={record['inputs']:4d}: "
            f"forward ratio={record['forward_ratio']:.6g}, "
            f"reverse ratio={record['reverse_ratio']:.6g}, "
            f"forward relative error={record['forward_relative_error']:.6e}, "
            f"reverse relative error={record['reverse_relative_error']:.6e}"
        )


if __name__ == "__main__":
    main()
