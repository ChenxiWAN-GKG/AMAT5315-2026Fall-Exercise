# Molecular dynamics examples

## Two-atom example

From the repository root, reproduce the two-panel relative-energy-error plot with:

```sh
cargo run --manifest-path week2/md/Cargo.toml --example dimer -- week2/dimer.png
```

The example uses the shared forward Euler and velocity-Verlet simulation and
converts its generated SVG to PNG with macOS Quick Look.

## Periodic Lennard-Jones fluid CLI

The `md` binary also runs a periodic two-dimensional Lennard-Jones fluid. The
contract configuration is:

```text
n = 100                 rho = 0.8
temperature = 0.5      dt = 0.01
eq_steps = 2000        steps = 10000
sample_every = 50      seed = 2026
```

Supported particle counts are even perfect squares, `n = s²`. The default run
uses a staggered triangular lattice, seeded Gaussian velocities, velocity-
Verlet integration, and rescaling only during equilibration. From the
repository root, run the complete workflow with:

```sh
cargo run --manifest-path week2/md/Cargo.toml --bin md -- run \
  --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 \
  --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 \
  --out artifacts
cargo run --manifest-path week2/md/Cargo.toml --bin md -- check artifacts
cargo run --manifest-path week2/md/Cargo.toml --bin md -- video artifacts \
  --out artifacts/run.mp4
```

The same defaults can be selected with `md run --out artifacts` after building
the binary. The run writes two files:

- `run.json` contains exactly `n`, `rho`, `box` (`[Lx,Ly]`), `dt`,
  `temperature`, `eq_steps`, `steps`, `sample_every`, `seed`, and
  `integrator` (`"velocity-verlet"`).
- `traj.jsonl` contains one JSON object per saved production frame. Each line
  has `step`, `t`, `pos`, `vel`, `E_pot`, and `E_kin`; `pos` and `vel` are
  arrays of `n` two-element pairs. Frames are saved when `step` is a positive
  multiple of `sample_every`, so the contract run has 200 frames at steps 50
  through 10000 and no frame-zero record.

`md check` recomputes energies from the saved positions and velocities without
advancing the simulation. It reports the stored-energy cross-check diagnostic,
`T_speed`, `abs(T_speed - 0.5)`, and the speed-shape statistic. The acceptance
limits are:

```text
drift < 2e-3
abs(T_speed - 0.5) < 0.05
chi_squared_22 < 2
```

The RDF uses minimum-image distances through
`min(Lx,Ly)/2`, counts both neighbour directions, averages over atoms and
saved frames, and normalizes by
`rho*pi*((r+dr)^2-r^2)`. The video is a deterministic 640×360 RGB movie with
atoms on the left and the cumulative RDF on the right, with one frame per
saved trajectory frame. It requires `ffmpeg` with the `libx264` encoder and
rejects output that is not smaller than 2,000,000 bytes.
