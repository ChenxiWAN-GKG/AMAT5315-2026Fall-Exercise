import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from compare_ad_modes import compare_modes


def test_modes_comparison_has_required_accuracy_and_plot(tmp_path):
    plot_path = tmp_path / "modes.png"
    errors = compare_modes(plot_path)

    assert plot_path.exists()
    assert errors["forward"] < 1e-12
    assert errors["reverse"] < 1e-12
    assert errors["finite_difference"] > errors["forward"]
    assert errors["finite_difference"] > errors["reverse"]
