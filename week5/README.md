# Week 5: automatic differentiation and checkpointed seismic imaging

The JAX scripts compare hand-written forward and reverse derivatives of a Lennard-Jones pair energy. The Rust `seismic` program propagates acoustic waves, differentiates each cell update with Enzyme, and composes those derivatives through time. Its adjoint can retain the whole forward trajectory or replay it with the binomial [Treeverse schedule](https://github.com/GiggleLiu/TreeverseAlgorithm.jl/blob/master/src/treeverse.jl). `seismic.design.toml` describes the command inputs and outputs.

From `week5/`, download the fixed survey inputs (which remain untracked) and use the course Python environment:

```sh
curl -fLO https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip
unzip -o week5-inputs.zip
rm week5-inputs.zip
~/.venvs/AMAT5315/bin/python scripts/lennard_jones_ad.py
~/.venvs/AMAT5315/bin/python scripts/compare_ad_modes.py
~/.venvs/AMAT5315/bin/python scripts/scale_cluster_ad.py
~/.venvs/AMAT5315/bin/python scripts/plot_graphs.py
~/.venvs/AMAT5315/bin/python scripts/plot_inputs.py
```

The saved 64-bit JAX example at `r=1.3` has pair energy −0.65701691446 and derivative 2.23997992979 in both hand-written modes and `jax.grad`. Across 601 separations from 0.95 to 2.5, maximum derivative errors are 2.13×10⁻¹⁴ (forward), 1.42×10⁻¹⁴ (reverse), and 4.95×10⁻⁹ (centered finite difference at `h=10⁻⁶`). The gradient jaxpr has an `add_any` operation joining the two contributions to the shared intermediate `a=r⁻⁶`; the graph figure is drawn from the operations JAX records.

The Rust crate pins `nightly-2026-09-05` with Enzyme. Install it from the crate directory, where `rust-toolchain.toml` takes effect:

```sh
cd seismic
cargo test --offline
cargo install --path . --locked
cd ..
seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
seismic --experiment inputs/reflector.json --mode born --out artifacts/born
seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --every 3 --out artifacts/adjoint
~/.venvs/AMAT5315/bin/python scripts/plot_forward.py --experiment inputs/reflector.json --traces artifacts/forward/traces.npy --out artifacts/forward/gathers.png
~/.venvs/AMAT5315/bin/python scripts/plot_results.py
```

The reflector's forward traces have L2 norm 11.574770. The Born data and image satisfy the transpose check `||d||² = ⟨m,Jᵀd⟩ = 0.03484789022`, with relative difference 1.99×10⁻¹⁶. The image's strongest depth row in the specified window is row 21, the reflector's true depth of 2.1 km. At step 150, the recorded echo peak is 0.0062247 versus 0.255243 for the background wave, a ratio of 2.44%.

For the three recorded field PNGs, open the [course seismic viewer](https://giggleliu.github.io/AMAT5315-2026Fall/week5-viewer.html) and select each `.npy` together with its folder's `run.json`. Save `forward/wavefield.npy` and `forward/echo.npy` at step 150 (3.00 s) as `artifacts/forward/wavefield.png` and `echo.png`; save `adjoint/wavefield.npy` at step 132 (2.64 s) as `artifacts/adjoint/wavefield.png`. The viewer reads times in reduced units from `run.json` and displays seconds using `time_unit_s`.

Reproduce the checkpoint comparison and Marmousi survey without a full-history Marmousi adjoint:

```sh
for b in 1 3 5 10; do
  seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --storage treeverse --checkpoints "$b" --out "artifacts/checkpoint-$b"
done
seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
seismic --experiment inputs/marmousi.json --mode adjoint --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 --out artifacts/marmousi-image
~/.venvs/AMAT5315/bin/python scripts/plot_checkpoint.py
```

Each Treeverse restart state contains both the previous and current pressure fields. The scheduler saves at most one initial state plus its additional-slot budget; `restore` resets the working state and its step index, `call` replays a forward step with the original source pulse, and `grad` applies one Enzyme reverse cell update using a saved state. Replay never adds to the image. The script audits the reverse order, valid restores, and slot counts for every shot.

| Additional slots | Forward calls per reflector shot | Peak states | Relative image error |
| ---: | ---: | ---: | ---: |
| 1 | 28,680 | 2 | 0 |
| 3 | 1,695 | 4 | 0 |
| 5 | 990 | 6 | 0 |
| 10 | 642 | 11 | 0 |
| Full history | 240 | 241 | Reference |

The action audit reports zero out-of-order or missing gradients, invalid restores, and budget overruns at every budget. Full history retains 6,481,936 bytes per reflector shot. Marmousi uses six saved states, or 20,788,320 bytes, instead of approximately 4.16 GB for full history of one shot. Its raw image L2 norm is 6.703774060×10⁻⁴. The plot shows curved and dipping shallow layers, while the deeper image remains weak on a single raw amplitude scale. The image differs from the velocity perturbation because it is the band-limited, survey-dependent normal-operator response `JᵀJm`.

## Evidence inventory

The commands above regenerate the committed evidence files. The `.npy` arrays and `inputs/` stay local; `MARMOUSI-LICENSE` accompanies the downloaded model.

| File | Producing command or script | Quantity shown |
| --- | --- | --- |
| `artifacts/ad/derivatives.json` | `scripts/lennard_jones_ad.py` | Node values, tangents, adjoints, JAX derivative |
| `artifacts/ad/modes.png` | `scripts/compare_ad_modes.py` | Derivative curves and absolute errors |
| `artifacts/ad/graph.png`, `grad-graph.png` | `scripts/plot_graphs.py` | JAX primal and gradient operations |
| `artifacts/ad/scaling.png` | `scripts/scale_cluster_ad.py` | Forward and reverse cluster-gradient cost |
| `artifacts/inputs.png` | `scripts/plot_inputs.py` | Survey geometry and Ricker pulse |
| `artifacts/forward/run.json`, `result.json` | `seismic --mode forward ...` | Experiment metadata and shape |
| `artifacts/forward/wavefield.png`, `echo.png` | Course viewer Save PNG at step 150 | Direct field and echo with frame readouts |
| `artifacts/forward/gathers.png` | `scripts/plot_forward.py ...` | Receiver pressure traces |
| `artifacts/born/run.json`, `result.json` | `seismic --mode born ...` | Born experiment metadata |
| `artifacts/adjoint/run.json`, `result.json` | `seismic --mode adjoint ...` | Full-history metadata and storage counters |
| `artifacts/adjoint/wavefield.png` | Course viewer Save PNG at step 132 | Reverse field with frame readout |
| `artifacts/adjoint/image.png` | `scripts/plot_results.py` | Reflector depth profile |
| `artifacts/checkpoint-{1,3,5,10}/run.json`, `result.json`, `actions-{0,1,2}.json` | Reflector checkpoint loop | Per-shot action schedule and storage/work counters |
| `artifacts/checkpoint-actions.png`, `checkpoint-work.png` | `scripts/plot_checkpoint.py` | Budget-5 action order and storage–work curve |
| `artifacts/marmousi-born/run.json`, `result.json` | Marmousi Born command | Large-survey metadata |
| `artifacts/marmousi-image/run.json`, `result.json`, `actions-{0…8}.json` | Marmousi checkpoint command | Six-state schedule and counters |
| `artifacts/marmousi.png` | `scripts/plot_checkpoint.py` | Background, perturbation, shot gather, raw image |

The `seismic` crate tests cover the wave step, Born and adjoint commands, the local Enzyme derivative identity, and the checkpoint schedule. The JAX and plotting tests are in `tests/`. A correct transpose identity checks the implemented discrete derivative; it does not make the finite-bandwidth migration image equal to the input perturbation.
