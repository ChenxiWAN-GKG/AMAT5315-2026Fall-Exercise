import numpy as np
import pytest

from scripts.chi_bootstrap import block_statistics, fit_peak


def test_fit_peak_returns_vertex_for_downward_parabola():
    temperatures = np.arange(2.2, 2.5, 0.05)
    susceptibilities = 10.0 - 40.0 * (temperatures - 2.3) ** 2

    peak, coefficients = fit_peak(temperatures, susceptibilities)

    assert peak == pytest.approx(2.3)
    assert coefficients[0] < 0.0


def test_fit_peak_rejects_vertex_outside_five_points():
    temperatures = np.array([2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5])
    susceptibilities = np.array(
        [8.55912294, 2.20969164, 1.66583177, 9.15536404, 1.57251889, 7.57206326, 3.12517016]
    )

    with pytest.raises(ValueError, match="outside"):
        fit_peak(temperatures, susceptibilities)


def test_block_statistics_uses_complete_blocks_only():
    stats = block_statistics(np.arange(5.0), 2)

    assert stats.shape == (2, 3)
    assert stats[:, 0].tolist() == [1.0, 5.0]
