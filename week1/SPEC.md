`estimate_pi(n, seed)` throws `n` random darts at the unit square (corners (0, 0) to (1, 1)), counts how many land within distance 1 of the origin, and returns four times that fraction.
The same `seed` must produce the same result.
The result is considered correct only when `abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2`