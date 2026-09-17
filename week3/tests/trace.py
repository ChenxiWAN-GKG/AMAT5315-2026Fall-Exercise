import numpy as np

from scripts.trace import first_sweeps


def test_first_sweeps_returns_absolute_magnetization_and_original_sweeps():
    records = [
        {"sweep": 7, "M": -0.5},
        {"sweep": 8, "M": 0.25},
        {"sweep": 9, "M": -0.75},
    ]

    sweeps, magnetization = first_sweeps(records, 2)

    assert np.array_equal(sweeps, [7, 8])
    assert np.allclose(magnetization, [0.5, 0.25])
