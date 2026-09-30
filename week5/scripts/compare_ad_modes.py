"""Compare hand-written AD modes with a centered finite difference."""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def energy(r: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Lennard-Jones energy through its named graph nodes."""
    a = r**-6
    b = a**2
    c = b - a
    return 4 * c


def analytic_derivative(r: jnp.ndarray) -> jnp.ndarray:
    """Return the exact derivative of 4(r^-12-r^-6)."""
    return 24 * (r**-7 - 2 * r**-13)


def forward_derivative(r: jnp.ndarray) -> jnp.ndarray:
    """Propagate a unit tangent one graph node at a time."""
    r_tangent = jnp.ones_like(r)
    a = r**-6
    a_tangent = (-6 * r**-7) * r_tangent
    b = a**2
    b_tangent = (2 * a) * a_tangent
    c_tangent = b_tangent - a_tangent
    return 4 * c_tangent


def reverse_derivative(r: jnp.ndarray) -> jnp.ndarray:
    """Propagate adjoints backward, adding both contributions to a."""
    a = r**-6
    c_adjoint = jnp.ones_like(r) * 4
    b_adjoint = c_adjoint
    a_adjoint = c_adjoint * (-1) + b_adjoint * (2 * a)
    return a_adjoint * (-6 * r**-7)


def finite_difference(r: jnp.ndarray, h: float = 1e-6) -> jnp.ndarray:
    """Estimate the derivative with the required centered difference."""
    return (energy(r + h) - energy(r - h)) / (2 * h)


def compare_modes(output_path: str | Path) -> dict[str, float]:
    """Make the comparison plot and return each method's maximum error."""
    r = jnp.linspace(0.95, 2.5, 601, dtype=jnp.float64)
    exact = analytic_derivative(r)
    forward = forward_derivative(r)
    reverse = reverse_derivative(r)
    finite = finite_difference(r)

    errors = {
        "forward": float(jnp.max(jnp.abs(forward - exact))),
        "reverse": float(jnp.max(jnp.abs(reverse - exact))),
        "finite_difference": float(jnp.max(jnp.abs(finite - exact))),
    }

    r_plot = r.tolist()
    exact_plot = exact.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(r_plot, exact_plot, label="analytic", linewidth=2)
    axes[0].plot(r_plot, forward.tolist(), "--", label="forward AD")
    axes[0].plot(r_plot, reverse.tolist(), ":", label="reverse AD")
    axes[0].plot(r_plot, finite.tolist(), "-.", label="finite difference")
    axes[0].set_xlabel("Separation r")
    axes[0].set_ylabel("dU/dr")
    axes[0].set_title("Derivative methods")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(r_plot, jnp.abs(forward - exact).tolist(), label="forward AD")
    axes[1].semilogy(r_plot, jnp.abs(reverse - exact).tolist(), label="reverse AD")
    axes[1].semilogy(r_plot, jnp.abs(finite - exact).tolist(), label="finite difference")
    axes[1].set_xlabel("Separation r")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("Absolute error")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return errors


def main() -> None:
    errors = compare_modes("artifacts/ad/modes.png")
    for method, error in errors.items():
        print(f"{method} maximum error: {error:.16e}")


if __name__ == "__main__":
    main()
