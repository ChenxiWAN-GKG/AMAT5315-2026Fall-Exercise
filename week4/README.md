# Week 4: time integration of a periodic flow

The `field` command writes one velocity field as JSON. The `fluid` command reads it from standard input, advances vorticity with Euler, midpoint (`rk2`), or classical RK4, and writes a tab-separated energy and enstrophy trace plus six-decimal field snapshots. The solver uses a periodic Fourier grid and keeps only modes with `|kx|, |ky| ≤ floor(n/3)` in both vorticity and nonlinear products. `field.design.toml` and `fluid.design.toml` specify the command interfaces.

From `week4/`, build and install the Rust tools, then regenerate the baseline runs:

```sh
cargo test
cargo build --release --bins
cargo install --path . --locked
mkdir -p artifacts evidence
field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green > artifacts/taylor-green.tsv
field taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json
field random --n 128 --seed 2026 --k-min 2 --k-max 6 | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --out artifacts/random > artifacts/random.tsv
cargo run --release --bin derivative-data
```

The derivative comparison uses `g = sin(3x) cos(2y)`. Maximum absolute errors were:

| Derivative | Centered N=32 | Centered N=64 | Fourier N=32 |
| --- | ---: | ---: | ---: |
| ∂x g | 0.17050404 | 0.04318456 | 1.30×10⁻¹⁴ |
| ∂xx g | 0.25724244 | 0.06487060 | 1.32×10⁻¹³ |
| ∂xy g | 0.48533864 | 0.12429411 | 3.64×10⁻¹⁴ |
| ∇²g | 0.30838312 | 0.07770515 | 1.48×10⁻¹³ |

The centered-difference errors shrink by factors of 3.90–3.97 when the grid spacing halves, close to the second-order factor of four. At `t=1`, Taylor–Green gives `E=0.167580`, `Z=0.335160`; the relative velocity error against the analytic field is `7.04×10⁻⁷` using the saved six-decimal frame. The random run starts at `E=0.500000`, `Z=6.634685` and ends at `E=0.286018`, `Z=0.955552` at `t=10`: enstrophy falls by a factor of 6.94 while energy falls by 42.8%. The initial maximum speed is 2.329828.

Regenerate the evidence figures with the course Python environment:

```sh
/Users/chenxi/.venvs/AMAT5315/bin/python scripts/plot_line_stability.py
/Users/chenxi/.venvs/AMAT5315/bin/python scripts/plot_line_accuracy.py
/Users/chenxi/.venvs/AMAT5315/bin/python scripts/flow_evidence.py
```

The plotting scripts need NumPy and Matplotlib. Replace the absolute interpreter path with another Python environment containing those packages if necessary. `flow_evidence.py` runs the stability scans, perturbation pairs, and refinement series itself. It stores their frames in untracked `artifacts/`.

| Committed evidence | Regeneration command | What it shows |
| --- | --- | --- |
| `evidence/line-stability.png` | `python scripts/plot_line_stability.py` | Measured RK4 stability region and line pulse |
| `evidence/line-accuracy.png` | `python scripts/plot_line_accuracy.py` | Line profiles and integrator error slopes |
| `evidence/taylor-green.png` | `python scripts/flow_evidence.py` | Vorticity and arrows at `t=0,1` on a shared scale |
| `evidence/random.png` | `python scripts/flow_evidence.py` | Seeded random vorticity at `t=0,2,5,10` |
| `evidence/blowup.png` | `python scripts/flow_evidence.py` | Energy and measured stopping times across stability limits |
| `evidence/sensitivity.png` | `python scripts/flow_evidence.py` | Relative vorticity separation after the specified ripple |
| `evidence/order.png` | `python scripts/flow_evidence.py` | Taylor–Green RK4 temporal order |
| `evidence/convergence.json` | `python scripts/flow_evidence.py` | Random-flow errors, fit, Richardson estimate, and choice |
| `evidence/convergence.png` | `python scripts/flow_evidence.py` | Random-flow time-step refinement and choice |

For `N=64`, the retained spectral corner has `|k|²=882`, giving an RK4 diffusion step limit `2.785/(0.1×882)=0.0316`. The Taylor–Green run at `Δt=0.032` stays finite through `t=8`; `Δt=0.033` stops at `t=7.82`. For the random field, the frozen-velocity estimate is `2.83/(2.329828×42√2)=0.0204`. The measured RK4 bracket is `Δt=0.030` (reaches `t=10`) and `0.035` (stops at `t=1.68`); Euler at `0.01` stops at `t=1.15`. The pairwise ripple causes relative vorticity separation to grow from `1.98×10⁻⁵` to `9.21×10⁻⁴` in the random flow over `t=0…20`, while it shrinks from `3.50×10⁻⁵` to `4.51×10⁻⁶` in Taylor–Green.

The Taylor–Green RK4 order fit at `N=8`, `ν=0.5`, `t=2` is 4.104. For the random flow at `N=128`, `ν=0.004`, `t=2`, the self-convergence slope against a `Δt=0.0025` reference is 4.027. Richardson estimation from `Δt=0.02` and `0.01` selects `Δt=0.0125`: predicted relative vorticity error `4.02×10⁻⁶`, measured `3.95×10⁻⁶`, both under the `5×10⁻⁶` target. These are time-discretization comparisons on the same grid, not spatial-convergence estimates.

The tests in `tests/` check the integrators, the represented-wave derivatives, and Taylor–Green decay. The plots and their underlying local trajectories support the additional visual and stability observations. The command tools are a periodic two-dimensional teaching model; they do not model walls or three-dimensional flow.
