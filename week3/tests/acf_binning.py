import numpy as np
import pytest

from scripts.acf_binning import autocorrelation, block_error


def test_autocorrelation_is_one_at_zero_lag():
    values = np.array([0.0, 1.0, 0.0, -1.0])

    result = autocorrelation(values)

    assert result[0] == pytest.approx(1.0)


def test_block_length_one_matches_naive_standard_error():
    values = np.array([1.0, 2.0, 4.0, 8.0])

    error, count = block_error(values, 1)

    assert error == pytest.approx(np.std(values, ddof=1) / np.sqrt(len(values)))
    assert count == len(values)
