import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from plot_inputs import plot_inputs


def test_input_plot_is_created(tmp_path):
    output = tmp_path / "inputs.png"
    plot_inputs(Path("inputs/reflector.json"), output)
    assert output.exists()
    assert output.stat().st_size > 10_000
