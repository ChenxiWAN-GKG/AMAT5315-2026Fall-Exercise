import random


def estimate_pi(n, seed):
    """Estimate pi by sampling ``n`` points from the unit square."""
    rng = random.Random(seed)
    inside_quarter_circle = sum(
        rng.random() ** 2 + rng.random() ** 2 <= 1 for _ in range(n)
    )
    return 4 * inside_quarter_circle / n
