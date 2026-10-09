import numpy as np

from scripts.chi_bootstrap import bootstrap_fits


def test_peak_bootstrap_resamples_each_temperature_independently():
    temperatures = np.arange(2.2, 2.41, 0.05)
    side = 8
    data = {}
    for temperature in temperatures:
        target_chi = 10.0 - 300.0 * (temperature - 2.3) ** 2
        amplitude = np.sqrt(target_chi * temperature / (side * side * 0.25))
        blocks = np.repeat([0.0] * 5 + [amplitude] * 5, 4)
        data[round(float(temperature), 2)] = blocks

    peaks, _, failed = bootstrap_fits(data, side, block_length=4, seed=2026)

    assert len(peaks) > 100
    assert failed < 400
    assert np.std(peaks) > 0.002
