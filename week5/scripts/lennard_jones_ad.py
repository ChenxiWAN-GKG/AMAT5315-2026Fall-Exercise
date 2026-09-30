"""Hand-written forward and reverse AD passes for a Lennard-Jones pair."""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


R = 1.3


def pair_energy(r: jnp.ndarray) -> jnp.ndarray:
    """Evaluate U(r) through the four computational-graph nodes."""
    a = r ** -6
    b = a**2
    c = b - a
    return 4 * c


def compute_derivatives(output_path: str | Path) -> dict[str, object]:
    """Compute and save the hand-written forward and reverse passes."""
    r = jnp.float64(R)

    # Forward mode: seed r with a unit tangent and apply one local JVP per node.
    r_tangent = jnp.float64(1.0)
    a = r**-6
    a_tangent = (-6 * r**-7) * r_tangent
    b = a**2
    b_tangent = (2 * a) * a_tangent
    c = b - a
    c_tangent = b_tangent - a_tangent
    energy = 4 * c
    energy_tangent = 4 * c_tangent

    # Reverse mode: seed U with one adjoint and apply one local VJP per node.
    energy_adjoint = jnp.float64(1.0)
    c_adjoint = energy_adjoint * 4
    b_adjoint = c_adjoint * 1
    a_adjoint = c_adjoint * (-1) + b_adjoint * (2 * a)
    r_adjoint = a_adjoint * (-6 * r**-7)

    jax_grad = jax.grad(pair_energy)(r)
    result = {
        "r": float(r),
        "energy": float(energy),
        "tangents": {
            "r": float(r_tangent),
            "a": float(a_tangent),
            "b": float(b_tangent),
            "c": float(c_tangent),
            "U": float(energy_tangent),
        },
        "adjoints": {
            "r": float(r_adjoint),
            "a": float(a_adjoint),
            "b": float(b_adjoint),
            "c": float(c_adjoint),
            "U": float(energy_adjoint),
        },
        "jax_grad": float(jax_grad),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    compute_derivatives(Path("artifacts/ad/derivatives.json"))


if __name__ == "__main__":
    main()
