# Seismic forward-mode package design

## Goal

Build a Rust `seismic` binary in `week5/seismic/` that follows
`week5/seismic.design.toml` and implements `forward` mode only. The binary
simulates the damped acoustic wave equation on a two-dimensional grid and
writes the specified JSON and NPY evidence files.

## Package structure

```text
seismic/
├── Cargo.toml
└── src/
    ├── main.rs
    ├── experiment.rs
    └── solver.rs
```

- `main.rs`: CLI parsing, mode dispatch, output directories, metadata, and
  process-level errors.
- `experiment.rs`: serde input structures, JSON loading, and validation.
- `solver.rs`: grid arrays, sponge construction, source evaluation, time
  stepping, receiver sampling, and optional recording.

Use standard crates: `clap` for the CLI, `serde`/`serde_json` for JSON,
`ndarray` for grid and trace arrays, and an NPY-compatible crate for the
required NumPy files. The numerical code remains explicit, beginner-readable
Rust loops.

## CLI and forward data flow

The supported command is:

```text
seismic --experiment inputs/reflector.json --mode forward \
        [--every N] --out artifacts/forward
```

The program will:

1. Load and validate the experiment JSON.
2. Construct the sponge field `sigma[z,x]` using the nearest-edge distance.
3. Run each shot independently with `u^-1 = 0` and `u^0 = 0`.
4. Advance the pressure field for exactly `steps` updates.
5. Sample receivers from `u^(n+1)` after each update.
6. Optionally retain first-shot frames at steps `0, every, 2*every, ...` and
   the final frame, according to the TOML recording convention.
7. For the first shot, optionally run the background-plus-perturbation model
   and save its difference from the background recording as `echo.npy`.
8. Write `run.json`, `result.json`, and `traces.npy`; write recording files
   only when `--every` is provided.

Only `forward` is accepted initially. Other modes produce a clear error.

## Numerical method

For interior cells, use the five-point Laplacian:

```text
L(u)[z,x] = (u[z,x-1] + u[z,x+1] + u[z-1,x] + u[z+1,x]
             - 4*u[z,x]) / dx^2
```

Use centered time differences and solve the damped equation as:

```text
u_next = (2*u - (1 - sigma*dt)*u_previous
          + dt^2*(c^2*L(u) + source)) / (1 + sigma*dt)
```

Boundary cells remain zero. The source at step `n` uses `t = n*dt` and is
`source_amplitude * Ricker(t) * Gaussian(x,z)`, where the Ricker pulse uses
`f0` and `t0`, and the Gaussian has unit peak and standard deviation one grid
cell. Coordinates are `[x,z]` in the input, while arrays are indexed `[z,x]`.

## Output contract

- `traces.npy`: float64 array `[shot, step, receiver]`.
- `wavefield.npy`: float32 first-shot frames when recording is enabled.
- `echo.npy`: float32 difference between the perturbed and background first-shot
  frames when recording is enabled.
- `run.json`: input path, experiment fields excluding large background and
  perturbation arrays, and recording steps/times when applicable.
- `result.json`: mode, grid dimensions, time-stepping values, shots, and
  receivers.

Inputs and all `.npy` files remain ignored by Git. JSON and PNG evidence may
be committed.

## Testing and verification

Unit tests will cover sponge values, Ricker/Gaussian source behavior, boundary
preservation, a hand-computed one-step update, and receiver sampling after an
update. Integration testing will run the forward command on the supplied
reflector experiment and verify array shapes, metadata, and the learning
sheet's reference trace norm and peak amplitudes within the stated tolerance.

The implementation will not add born, adjoint, or checkpointing behavior; those
are later stages of the learning sheet.
