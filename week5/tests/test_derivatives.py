import json
import math
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from lennard_jones_ad import compute_derivatives


def test_derivatives_json_values_and_nodes(tmp_path):
    output = tmp_path / "derivatives.json"
    result = compute_derivatives(output)

    saved = json.loads(output.read_text())
    assert saved == result
    assert math.isclose(saved["r"], 1.3, abs_tol=1e-15)
    assert math.isclose(saved["energy"], -0.6570169144600471, abs_tol=1e-12)

    assert set(saved["tangents"]) == {"r", "a", "b", "c", "U"}
    assert set(saved["adjoints"]) == {"r", "a", "b", "c", "U"}
    assert math.isclose(saved["tangents"]["r"], 1.0, abs_tol=1e-15)
    assert math.isclose(saved["tangents"]["U"], 2.239979929791143, abs_tol=1e-12)
    assert math.isclose(saved["adjoints"]["r"], 2.239979929791143, abs_tol=1e-12)
    assert math.isclose(saved["adjoints"]["a"], -2.3425903117359734, abs_tol=1e-12)
    assert math.isclose(saved["jax_grad"], saved["adjoints"]["r"], abs_tol=1e-12)

    assert jax.config.read("jax_enable_x64")
