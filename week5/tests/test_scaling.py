import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from scale_cluster_ad import run_scaling


def test_cluster_scaling_writes_plot_and_checks_forces(tmp_path):
    records = run_scaling(tmp_path / "scaling.png", sizes=(8,))

    assert (tmp_path / "scaling.png").exists()
    assert len(records) == 1
    record = records[0]
    assert record["inputs"] == 24
    assert record["forward_ratio"] > 0
    assert record["reverse_ratio"] > 0
    assert record["forward_relative_error"] < 1e-12
    assert record["reverse_relative_error"] < 1e-12
