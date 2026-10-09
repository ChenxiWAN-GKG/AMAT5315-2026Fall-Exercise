import json

import numpy as np
import pytest

from scripts.compare import read_window, work_normalized_tau


def test_read_window_keeps_magnetization_and_cluster_sizes(tmp_path):
    (tmp_path / "run.json").write_text(
        json.dumps({"L": 4, "update": "wolff", "measure": 2}), encoding="utf-8"
    )
    rows = [
        {"L": 4, "T": 2.3, "sweep": 1, "M": -0.5, "E": -1.0, "cluster_size": 3},
        {"L": 4, "T": 2.3, "sweep": 2, "M": 0.25, "E": -1.2, "cluster_size": 5},
    ]
    (tmp_path / "series.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    samples = read_window(tmp_path, 4, "wolff")

    assert np.array_equal(samples[2.3][0], [-0.5, 0.25])
    assert np.array_equal(samples[2.3][1], [3, 5])


def test_work_normalized_tau_counts_flipped_spins():
    assert work_normalized_tau(4.0, 1024.0, 64) == pytest.approx(1.0)
