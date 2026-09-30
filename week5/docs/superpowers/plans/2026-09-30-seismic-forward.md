# Seismic Forward Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust `seismic` binary that implements the learning sheet's `forward` acoustic-wave mode and writes its specified JSON and NPY outputs.

**Architecture:** A small Cargo binary will separate input parsing/validation, numerical stepping, and CLI/output orchestration. The solver will use explicit `ndarray` arrays indexed `[z, x]`, while `main.rs` will run shots, collect receiver traces, and write metadata and optional recordings.

**Tech Stack:** Rust 2021, `clap`, `serde`, `serde_json`, `ndarray`, `ndarray-npy`, Cargo, and Rust's built-in test framework.

**Spec:** `docs/seismic-forward-design.md` and `seismic.design.toml`

## Global Constraints

- Work only in `week5/`; preserve existing AD artifacts and unrelated untracked files.
- Implement `forward` mode only; reject `born` and `adjoint` with a clear error.
- Use `u^-1 = 0` and `u^0 = 0`; do not use a half-step initialization.
- Keep outer grid cells zero and update only interior cells.
- Use the five-point Laplacian and the exact damped update from the design document.
- Evaluate the source at `t = n * dt` and sample receivers from `u^(n+1)`.
- Use `[z, x]` for arrays even though input shot/receiver coordinates are `[x, z]`.
- Keep `inputs/` and all `.npy` files out of Git.
- Put run/comparison/plotting scripts in `scripts/`; this Rust task needs no new plotting script.
- Run tests with Cargo and commit each completed task separately.

## File Map

- Create `seismic/Cargo.toml`: binary package and dependency declarations.
- Create `seismic/src/lib.rs`: public module declarations for unit testing.
- Create `seismic/src/experiment.rs`: serde input types and validation.
- Create `seismic/src/solver.rs`: sponge, source, Laplacian, timestep, receiver sampling, and recording-frame helpers.
- Create `seismic/src/main.rs`: CLI, shot loop, output metadata, NPY writes, and process errors.
- Create `seismic/tests/forward_integration.rs`: command-level output and shape checks using a temporary small experiment.

### Task 1: Create the Cargo package and validated experiment model

**Files:**
- Create: `seismic/Cargo.toml`
- Create: `seismic/src/lib.rs`
- Create: `seismic/src/experiment.rs`
- Test: `seismic/src/experiment.rs` unit tests

**Interfaces:**
- Produce `pub struct Experiment` with fields `nx`, `nz`, `dx`, `dt`, `steps`, `source_frequency`, `source_peak_time`, `source_amplitude`, `shots: Vec<[usize; 2]>`, `receivers: Vec<[usize; 2]>`, `sponge_width`, `sponge_strength`, `background: Array2<f64>`, `perturbation: Array2<f64>`, `length_unit_m`, and `time_unit_s`.
- Produce `pub fn load_experiment(path: &Path) -> Result<Experiment, String>`.
- Produce `Experiment::validate(&self) -> Result<(), String>` checking array shape, nonempty shots/receivers, positive dimensions and spacings, nonnegative steps, and in-bounds shot/receiver coordinates.

- [ ] **Step 1: Write failing validation tests**

Add tests for a valid small experiment, a background shape mismatch, and an out-of-bounds receiver. Use JSON strings written to `tempfile::NamedTempFile` and assert the returned error contains the relevant field name.

- [ ] **Step 2: Run the tests to verify failure**

Run:

```bash
cd seismic
cargo test experiment
```

Expected: compilation/test failure because the package and loader do not yet exist.

- [ ] **Step 3: Add the package and serde model**

Use `serde_json::from_reader` to parse the input object. Convert nested `Vec<Vec<f64>>` background and perturbation values to `Array2<f64>`, preserving `[z][x]` order. Add the dependencies:

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
ndarray = "0.16"
ndarray-npy = "0.9"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 4: Run the validation tests**

Run `cargo test experiment`; all validation tests must pass.

- [ ] **Step 5: Commit**

```bash
git add seismic/Cargo.toml seismic/src/lib.rs seismic/src/experiment.rs
git commit -m "Add seismic experiment model and validation"
```

### Task 2: Implement and test the acoustic numerical kernel

**Files:**
- Create: `seismic/src/solver.rs`
- Modify: `seismic/src/lib.rs`
- Test: `seismic/src/solver.rs` unit tests

**Interfaces:**
- Produce `pub fn build_sponge(nx: usize, nz: usize, width: f64, strength: f64) -> Array2<f64>`.
- Produce `pub fn ricker(t: f64, frequency: f64, peak_time: f64) -> f64`.
- Produce `pub fn gaussian_footprint(nx: usize, nz: usize, shot_x: usize, shot_z: usize) -> Array2<f64>`.
- Produce `pub fn advance(previous: &Array2<f64>, current: &Array2<f64>, next: &mut Array2<f64>, speed: &Array2<f64>, sigma: &Array2<f64>, source: &Array2<f64>, dx: f64, dt: f64)`.
- Produce `pub fn sample_receivers(field: &Array2<f64>, receivers: &[[usize; 2]]) -> Vec<f64>` where each coordinate is `[x, z]`.

- [ ] **Step 1: Write failing numerical tests**

Add tests that assert: edge sponge values equal `strength`, a point outside the sponge has zero damping, `ricker(peak_time, ...)` equals `1`, the Gaussian center equals `1`, a one-cell source produces the hand-computed one-step update, boundary cells remain zero, and receiver sampling maps `[x,z]` to `[z,x]` after the update.

- [ ] **Step 2: Run the tests to verify failure**

Run:

```bash
cd seismic
cargo test solver
```

Expected: failure because the solver functions are not implemented.

- [ ] **Step 3: Implement the kernel**

Use the exact formulas:

```rust
let laplacian = (current[[z, x - 1]]
    + current[[z, x + 1]]
    + current[[z - 1, x]]
    + current[[z + 1, x]]
    - 4.0 * current[[z, x]]) / (dx * dx);
next[[z, x]] = (2.0 * current[[z, x]]
    - (1.0 - sigma[[z, x]] * dt) * previous[[z, x]]
    + dt * dt * (speed[[z, x]].powi(2) * laplacian + source[[z, x]]))
    / (1.0 + sigma[[z, x]] * dt);
```

Set `next.fill(0.0)` before updating interior cells. Compute sponge distance as the minimum of `x`, `nx-1-x`, `z`, and `nz-1-z`; use `strength * max(0, 1 - distance / width)^2` and return zero when width is zero.

- [ ] **Step 4: Run the numerical tests**

Run `cargo test solver`; all kernel tests must pass.

- [ ] **Step 5: Commit**

```bash
git add seismic/src/lib.rs seismic/src/solver.rs
 git commit -m "Implement seismic acoustic forward kernel"
```

### Task 3: Add the forward CLI and required output files

**Files:**
- Create: `seismic/src/main.rs`
- Modify: `seismic/Cargo.toml` if needed
- Test: `seismic/tests/forward_integration.rs`

**Interfaces:**
- CLI flags: required `--experiment`, required `--mode`, optional `--every`, required `--out`.
- `forward` writes `run.json`, `result.json`, and `traces.npy`; with `--every`, it also writes first-shot `wavefield.npy` and `echo.npy`.
- `traces.npy` has shape `[shots, steps, receivers]` and float64 values.

- [ ] **Step 1: Write a failing integration test**

Create a temporary 5x5, two-step experiment with one shot and two receivers. Invoke the compiled binary with `std::process::Command`, then assert successful exit, required files, trace shape, and a zero outer boundary in the first recorded frame.

- [ ] **Step 2: Run the integration test to verify failure**

Run `cd seismic && cargo test --test forward_integration`; expected failure because no binary exists.

- [ ] **Step 3: Implement CLI orchestration**

For each shot, build the source footprint, allocate zero previous/current fields, run `steps` updates, call `sample_receivers` after each `advance`, and rotate arrays by swapping previous/current/next storage. Serialize traces with `ndarray_npy::write_npy`.

For recording, save frame 0 before updates and each frame whose step is divisible by `every`, plus the final step if not already saved. Run the first shot twice when recording: once with `background`, once with `background + perturbation`; store the frame-by-frame difference as float32 `echo.npy`.

Write compact metadata JSON. `run.json` must preserve the experiment path and scalar/list fields but omit the large background and perturbation arrays. `result.json` must contain mode, dimensions, time-step values, shots, and receivers.

Return a nonzero exit with a readable message for unsupported modes, invalid `--every 0`, malformed input, or output-write failures.

- [ ] **Step 4: Run integration and unit tests**

Run:

```bash
cd seismic
cargo test
```

All tests must pass, including the temporary CLI experiment.

- [ ] **Step 5: Commit**

```bash
git add seismic/src/main.rs seismic/tests/forward_integration.rs seismic/Cargo.toml
 git commit -m "Add seismic forward CLI and outputs"
```

### Task 4: Run the supplied reflector experiment and verify evidence

**Files:**
- Modify only generated local files under `artifacts/forward/`; `.npy` files remain ignored.
- Optional: Create `scripts/inspect_forward.py` only if a small plotting/inspection helper is needed; save any PNG under `artifacts/forward/`.

- [ ] **Step 1: Build and run the required command**

```bash
cd seismic
cargo run --release -- \
  --experiment ../inputs/reflector.json \
  --mode forward --every 3 --out ../artifacts/forward
```

- [ ] **Step 2: Inspect generated shapes and metadata**

Use the course Python environment and NumPy to load `traces.npy` and confirm shape `[3, 240, 14]`, float64 dtype, and recording frame shapes. Confirm JSON fields match the experiment.

- [ ] **Step 3: Compare against the learning-sheet values**

Compute the all-trace L2 norm and each shot's largest absolute pressure. Require all-trace norm relative error below `1e-4` against `11.574770`, and outer/central/outer maxima within `1e-4` relative error of `0.60809514`, `0.59271397`, and `0.60809514`.

- [ ] **Step 4: Commit reproducible evidence**

Commit JSON evidence and any PNG inspection plot, but not `inputs/` or `.npy` files:

```bash
git add seismic artifacts/forward/*.json artifacts/forward/*.png
 git commit -m "Verify seismic forward experiment"
```

## Plan self-review

- The plan covers package setup, JSON input validation, all specified numerical terms, receiver timing, optional recordings, output formats, unsupported-mode behavior, and the supplied forward verification.
- No born, adjoint, or checkpoint behavior is included.
- Every numerical test has an explicit expected behavior and every task ends with a test and commit.
- `inputs/` and `.npy` remain ignored; the existing Week 5 AD work is not modified.
