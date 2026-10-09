# Week 2: Lennard-Jones molecular dynamics

The Rust crate in `md/` uses a Lennard-Jones pair force, a staggered triangular initial lattice, periodic boundaries, and velocity-Verlet integration. `md.design.toml` records the simulation and file contract. Run the commands below from `week2/`. Generated unheated `artifacts/` data and Rust build files remain local; the 400-atom heating recording is committed for the course viewer.

Install Rust and Python with NumPy and Matplotlib, plus `ffmpeg` for video. Build the release and debug binaries with `cargo build --release --manifest-path md/Cargo.toml` and `cargo build --manifest-path md/Cargo.toml`. `make reproduce` builds and runs the unheated contract case into `artifacts/`:

```sh
make reproduce
python3 scripts/check.py artifacts | tee check.txt
python3 scripts/video.py artifacts fluid.mp4
```

The saved positions and velocities, rounded to nine significant digits, determine each stored energy. `scripts/check.py` independently recomputes the shifted pair potential and kinetic energy from those serialized values. The verified result for the seed-2026 run is a secular drift of 0.00011047 (limit 0.002), a pooled speed temperature of 0.508809 (distance from 0.5 is 0.008809, limit 0.05), and χ²/22 = 1.2930 for 24 equal-probability Rayleigh bins (limit 2). The largest stored-versus-recomputed energy difference is 5.12 × 10⁻¹³. These are unheated production measurements; the thermostat is off during those 10,000 steps.

The pair force is F(r) = 24(2r⁻¹² − r⁻⁶)/r, the negative slope of U(r) = 4(r⁻¹² − r⁻⁶). `force-comparison.txt` compares it with a central energy difference at four separations, using h = 10⁻⁵. All four absolute differences are below 10⁻⁶ max(1, |F|). `force-paths.txt` compares the naive and cell-list pair searches on a perturbed lattice, a boundary-crossing pair, a cutoff pair, and a two-cell-wide box. Their largest component force difference is 1.44 × 10⁻¹⁵; the largest energy difference is 5.68 × 10⁻¹⁴.

## Timing

Each entry is the median of three elapsed wall-clock runs, followed by the minimum–maximum range. The NumPy baseline is the [course script](https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week2-sim.py), run with the course Python environment; its fixed seed is 42, while the Rust contract uses seed 2026. All three runs have 100 atoms, 2,000 equilibration steps, and 10,000 production steps.

| Program | Median (s) | Range: min–max (s) |
| --- | ---: | ---: |
| NumPy `week2-sim.py` | 2.654 | 2.579–2.671 |
| Rust debug | 5.627 | 5.592–5.995 |
| Rust release | 0.494 | 0.488–0.659 |

The release median is 11.4 times faster than debug. These are wall-clock times on this machine, not a claim about other hardware.

## Profile

The inclusive force share counts samples in `forces_and_energy_with_method` and its calls. `profile-*.json.gz` hold the original `samply` profiles; the companion `.syms.json` files preserve function names. The two PNGs capture the `samply` call trees, and `scripts/profile.py` recomputes the sampled shares from the raw call stacks.

| Version | Force share (%) | Profiled run (s) |
| --- | ---: | ---: |
| Naive | 97.7 | 0.650 |
| Cell list | 92.9 | 0.253 |

The cell list restricts each atom to its cell and eight neighbours, while the naive loop tests every atom pair. The force share can remain high even when the absolute run gets faster because it is a fraction of the shorter total runtime.

## Benchmark

Each time covers 100 equilibration and 500 production steps in a release build. Parentheses give the range of three runs. Speedup divides the naive median by the cell-list median.

| N | Naive (s) | Cells (s) | Speedup |
| ---: | ---: | ---: | ---: |
| 100 | 0.028 (0.028–0.029) | 0.028 (0.028–0.028) | 1.00× |
| 400 | 0.310 (0.309–0.311) | 0.106 (0.105–0.107) | 2.91× |
| 1600 | 4.307 (4.063–4.458) | 0.475 (0.474–0.477) | 9.06× |

At fixed density, the naive candidate count grows roughly with N², while each cell contains a roughly bounded number of atoms. `scaling.png` shows the measured seconds per step; the table retains the exact medians and ranges.

## Heating and viewer

The heating trajectory raises the target temperature from 0.2 to 1.2 over production and rescales velocities every 50 steps. Recreate it with:

```sh
./md/target/release/md --n 400 --rho 0.8 --temperature 0.2 --ramp-to 1.2 --dt 0.01 --eq-steps 2000 --steps 20000 --sample-every 100 --seed 2026 --out heating
./md/target/release/md --n 100 --rho 0.8 --temperature 0.2 --dt 0.01 --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 --out /tmp/amat5315-week2-cold
./md/target/release/md --n 100 --rho 0.8 --temperature 1.0 --dt 0.01 --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 --out /tmp/amat5315-week2-hot
python3 scripts/video.py /tmp/amat5315-week2-cold cold.mp4
python3 scripts/video.py /tmp/amat5315-week2-hot hot.mp4
```

Both videos contain 200 frames. Verified visually at the final frame: the cold atoms remain near triangular lattice sites and have repeated long-range pair peaks, while the hot atoms have a flatter distant pair structure. `fluid-viewer.png` is the course viewer's stamped final frame of the unheated contract run. The [heating recording in the course viewer](https://giggleliu.github.io/AMAT5315-2026Fall/week2-viewer.html?src=https%3A%2F%2Fraw.githubusercontent.com%2FChenxiWAN-GKG%2FAMAT5315-2026Fall-Exercise%2Fmain%2Fweek2%2Fheating%2Ftraj.jsonl) can be opened without a local checkout after the committed files are pushed.

## Evidence inventory

| Evidence file | Generating command |
| --- | --- |
| `field.png` | `cargo run --manifest-path md/Cargo.toml --bin field -- field.png` |
| `dimer.png` | `cargo run --manifest-path md/Cargo.toml --example dimer -- dimer.png` |
| `force-comparison.txt` | `cargo run --release --manifest-path md/Cargo.toml --example force_comparison --quiet > force-comparison.txt` |
| `force-paths.txt` | `cargo run --release --manifest-path md/Cargo.toml --example force_paths --quiet > force-paths.txt` |
| `check.txt` | `python3 scripts/check.py artifacts > check.txt` after `make reproduce` |
| `fluid.mp4` | `python3 scripts/video.py artifacts fluid.mp4` |
| `fluid-viewer.png` | Load `artifacts/traj.jsonl` and `artifacts/run.json` in the course viewer, select frame 199, and save PNG |
| `benchmark.json`, `scaling.png` | Download the course `week2-sim.py`, then `python3 scripts/benchmark.py --baseline week2-sim.py` |
| `profile-naive.json.gz`, `profile-naive.json.syms.json` | `samply record --save-only --unstable-presymbolicate -o profile-naive.json.gz -- ./md/target/release/md --n 400 --rho 0.8 --temperature 0.5 --dt 0.01 --eq-steps 200 --steps 1000 --sample-every 50 --seed 2026 --force naive --out /tmp/md-prof` |
| `profile-cells.json.gz`, `profile-cells.json.syms.json` | Repeat the preceding command with `--force cells` and `-o profile-cells.json.gz` |
| `profile-naive.png`, `profile-cells.png` | Open the saved profiles with `samply load profile-naive.json.gz` and `samply load profile-cells.json.gz`, then capture each call tree |
| `heating/run.json`, `heating/traj.jsonl` | The 400-atom heating command above |
| `cold.mp4`, `hot.mp4` | The two fixed-temperature runs and video commands above |

In the Rust code, each `FluidState` owns its position and velocity vectors. An immutable borrow lets a force calculation read a state; a mutable borrow lets velocity-Verlet replace a state after its next position and velocity are computed. The shared `Integrator` trait in the two-atom example allows the same experiment driver to call Euler or Verlet. The plotted Euler error grows because it does not preserve the bounded energy behavior seen with Verlet for this pair. These terms describe Rust's data access rules, not a change to the physical force law.

## Limitations

The timing and profile measurements are machine-specific. The Lennard-Jones model is two-dimensional and uses velocity rescaling, so the videos illustrate this exercise's solid-to-fluid structural change rather than a general material's melting point.
