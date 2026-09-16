# Ising CLI Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with test-first checkpoints. Steps use checkbox syntax for tracking.

**Goal:** Build a seeded Rust CLI that samples the finite-temperature two-dimensional Ising model with both Metropolis and Wolff updates and writes the schemas in `ising.design.toml`.

**Architecture:** A small `ising` library will own the periodic lattice, energy/magnetization calculations, Metropolis sweeps, Wolff cluster flips, temperature-ramp execution, and output records. A thin `main.rs` will parse the exact command-line flags, reject missing or invalid values, call the library, and print the required temperature summary. JSON output will be written directly with fixed-point formatting so six-decimal fields retain six decimals.

**Tech Stack:** Rust 2024, Cargo, `rand = "0.9.5"` with `StdRng::seed_from_u64`, standard-library file I/O, and `serde_json` only as a test dependency for validating emitted JSON.

**Spec:** `ising.design.toml`

## Global Constraints

- The lattice side `l` is an integer at least 2, with periodic square-lattice neighbors.
- `J = 1`, no external field, and every temperature in the generated ramp is finite and strictly positive.
- `metropolis` performs exactly `l*l` independent uniformly selected site proposals per step, with replacement.
- `wolff` performs one cluster flip per step using bond probability `1 - exp(-2/T)`.
- The first temperature starts from an all-up lattice; each later temperature continues from the previous lattice.
- All flags except `--every` are required; `--every` defaults to 0; no other defaults are introduced.
- `series.jsonl` uses six decimal places for `T`, `M`, and `E`; Wolff records add integer `cluster_size`.
- `run.json` records `sample_every = 1` for the measured series; `--every` controls only optional spin frames.
- `spins.jsonl` is written only when `every > 0`, and its sweep counter is cumulative across discard and measurement steps and across the whole ramp.
- Existing files with names written by this run may be truncated; unrelated files in the output directory are preserved.
- Do not stage, commit, or modify files outside `week3/`.

---

### Task 1: Scaffold the crate and test the lattice invariants

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `tests/lattice.rs`

**Interfaces:**
- Produce `ising::Lattice::all_up(side: usize) -> Lattice`.
- Produce `Lattice::magnetization(&self) -> f64` and `Lattice::energy_per_site(&self) -> f64`.
- Produce `Lattice::delta_energy(&self, index: usize) -> i32` for flipping one site.
- Produce `Lattice::flip(&mut self, index: usize)`.

- [ ] **Step 1: Write the failing lattice tests**

```rust
use ising::Lattice;

#[test]
fn all_up_two_by_two_has_magnetization_one_and_energy_minus_two() {
    let lattice = Lattice::all_up(2);
    assert_eq!(lattice.magnetization(), 1.0);
    assert_eq!(lattice.energy_per_site(), -2.0);
}

#[test]
fn local_flip_energy_matches_the_total_energy_change() {
    let mut lattice = Lattice::all_up(3);
    let before = lattice.energy_per_site();
    let delta = lattice.delta_energy(0) as f64;
    lattice.flip(0);
    let after = lattice.energy_per_site();
    assert!((after - before) * 9.0 == delta);
}
```

- [ ] **Step 2: Run the new tests and confirm the missing-lattice failure**

Run: `cargo test --test lattice`

Expected: compilation fails because the new `ising` library and `Lattice` API do not exist yet.

- [ ] **Step 3: Add the minimal Cargo manifest and periodic lattice implementation**

Use `edition = "2024"`, package name `ising`, `rand = "0.9.5"` as a runtime dependency, and `serde_json = "1"` as a dev-dependency. Store spins row-major as `Vec<i8>`, count each of the four directional neighbor contributions (including the repeated neighbors that occur when `side == 2`), and use

```text
E/site = -sum_i(s_i * neighbor_sum_i) / (2 * side * side)
```

for the energy per site.

- [ ] **Step 4: Run the focused tests and the formatter**

Run: `cargo fmt --check && cargo test --test lattice`

Expected: both lattice tests pass.

### Task 2: Add seeded Metropolis and Wolff update kernels

**Files:**
- Modify: `src/lib.rs`
- Create: `tests/updates.rs`

**Interfaces:**
- Produce `Lattice::metropolis_step<R: rand::Rng + ?Sized>(&mut self, temperature: f64, rng: &mut R) -> usize`, returning accepted flips after exactly `side*side` proposals.
- Produce `Lattice::wolff_step<R: rand::Rng + ?Sized>(&mut self, temperature: f64, rng: &mut R) -> usize`, returning the number of spins flipped in the cluster.

- [ ] **Step 1: Write failing update tests**

```rust
use ising::Lattice;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn metropolis_step_reports_at_most_one_acceptance_per_proposal() {
    let mut lattice = Lattice::all_up(3);
    let mut rng = StdRng::seed_from_u64(2026);
    let accepted = lattice.metropolis_step(2.0, &mut rng);
    assert!(accepted <= 9);
}

#[test]
fn wolff_step_flips_a_nonempty_same_spin_cluster() {
    let mut lattice = Lattice::all_up(4);
    let before = lattice.magnetization();
    let mut rng = StdRng::seed_from_u64(2026);
    let cluster_size = lattice.wolff_step(2.0, &mut rng);
    assert!((1..=16).contains(&cluster_size));
    assert_eq!(lattice.magnetization(), before - 2.0 * cluster_size as f64 / 16.0);
}
```

- [ ] **Step 2: Run the update tests and confirm the missing-method failure**

Run: `cargo test --test updates`

Expected: compilation fails because the update methods do not exist yet.

- [ ] **Step 3: Implement the Metropolis kernel**

Select a site with `rng.random_range(0..self.spins.len())` for every one of `side*side` proposals, compute `delta_energy = 2*s_i*neighbor_sum_i`, accept negative or zero energy changes, and accept positive changes with probability `exp(-delta_energy as f64 / temperature)`. Count accepted flips.

- [ ] **Step 4: Implement the Wolff kernel**

Select one seed with `rng.random_range(0..self.spins.len())`, grow a stack-backed cluster through same-spin neighbors, test each directional bond with `1 - exp(-2.0 / temperature)`, mark sites when added, flip every marked site once, and return the cluster size. Keep the four directional attempts so periodic `side == 2` multiplicities match the energy definition.

- [ ] **Step 5: Run all current tests**

Run: `cargo fmt --check && cargo test`

Expected: the lattice and update tests pass.

### Task 3: Implement the temperature ramp and JSONL records

**Files:**
- Modify: `src/lib.rs`
- Create: `tests/ramp.rs`

**Interfaces:**
- Produce `RunConfig` with the parsed numerical settings and output path.
- Produce `run(config: &RunConfig) -> Result<Vec<TemperatureSummary>, SimulationError>`.
- Produce JSON records with the exact keys from `ising.design.toml`: `run.json`, `series.jsonl`, and optional `spins.jsonl`.

- [ ] **Step 1: Write failing ramp/output tests**

```rust
use std::path::{Path, PathBuf};

fn unique_test_directory(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ising-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn small_config(output: &Path, update: ising::Update) -> ising::RunConfig {
    ising::RunConfig {
        update,
        l: 2,
        t_from: 1.5,
        t_to: 1.6,
        t_step: 0.05,
        discard: 1,
        measure: 2,
        seed: 2026,
        every: 0,
        out: output.to_path_buf(),
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn run_writes_a_temperature_grid_and_resets_measured_sweeps() {
    let output = unique_test_directory("ramp");
    let config = small_config(&output, ising::Update::Metropolis);
    ising::run(&config).unwrap();

    let run_json: serde_json::Value = read_json(&output.join("run.json"));
    assert_eq!(run_json["t_grid"], serde_json::json!([1.5, 1.55, 1.6]));
    let series = read_lines(&output.join("series.jsonl"));
    assert_eq!(series.len(), 6);
    assert!(series[0].contains("\"sweep\":1"));
    assert!(series[2].contains("\"sweep\":1"));
}

#[test]
fn spin_frames_use_the_cumulative_counter_and_six_decimal_numbers() {
    let output = unique_test_directory("frames");
    let mut config = small_config(&output, ising::Update::Wolff);
    config.every = 2;
    ising::run(&config).unwrap();

    let frames = read_lines(&output.join("spins.jsonl"));
    assert!(frames.iter().all(|line| line.contains(".000000")));
    assert_eq!(frames.len(), 3);
    assert!(frames[0].contains("\"sweep\":3"));
    assert!(frames[1].contains("\"sweep\":6"));
    assert!(frames[2].contains("\"sweep\":9"));
}
```

- [ ] **Step 2: Run the ramp tests and confirm the missing-run failure**

Run: `cargo test --test ramp`

Expected: compilation fails because `RunConfig`, `run`, and the output writers do not exist yet.

- [ ] **Step 3: Implement validated temperature-grid construction and ramp state**

Require `t_to >= t_from > 0`, `t_step > 0`, and finite values. Build the grid by integer step index and include `t_to` only when it is reached by that index within a machine-precision comparison. Initialize one all-up lattice, run discard steps without series records, then record each measured step while continuing the same lattice into the next temperature.

- [ ] **Step 4: Implement the three output writers**

Write `run.json` once, write one `series.jsonl` object per measured step, and write `spins.jsonl` only for positive `every` after measured steps `every, 2*every, ...` at each temperature; store the cumulative discard-inclusive sweep in each frame. Format all floating-point values required by the spec with `format!("{:.6}", value)`. Use `File::create` for the named files and do not remove other output-directory contents.

- [ ] **Step 5: Run the ramp tests and inspect parsed output**

Run: `cargo fmt --check && cargo test --test ramp`

Expected: both ramp/output tests pass and every generated JSON line parses successfully.

### Task 4: Add the exact CLI and end-to-end coverage

**Files:**
- Create: `src/main.rs`
- Create: `tests/cli.rs`
- Modify: `Cargo.toml` only if the binary target needs explicit configuration

**Interfaces:**
- Parse `ising --update ... --l ... --t-from ... --t-to ... --t-step ... --discard ... --measure ... --every ... --seed ... --out ...`.
- Accept `metropolis` and `wolff`; reject unknown updates, missing required flags, non-finite/non-positive temperatures, `l < 2`, `measure == 0`, and unknown or malformed flags.
- Use `every = 0` only when `--every` is omitted.

- [ ] **Step 1: Write failing CLI tests**

```rust
#[test]
fn cli_runs_both_updates_and_emits_one_summary_line_per_temperature() {
    let output_path = unique_test_directory("cli");
    for update in ["metropolis", "wolff"] {
        let output = output_path.to_str().unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_ising"))
            .args(["--update", update, "--l", "2", "--t-from", "1.5",
                   "--t-to", "1.6", "--t-step", "0.05", "--discard", "1",
                   "--measure", "2", "--seed", "2026", "--out", output])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(String::from_utf8_lossy(&result.stdout).lines().count(), 4);
    }
}

#[test]
fn cli_rejects_zero_temperature_and_missing_required_flags() {
    let zero = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(["--update", "metropolis", "--l", "2", "--t-from", "0"])
        .output()
        .unwrap();
    assert!(!zero.status.success());
    let missing = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(["--update", "metropolis"])
        .output()
        .unwrap();
    assert!(!missing.status.success());
}
```

- [ ] **Step 2: Run CLI tests and confirm the missing-binary/parser failure**

Run: `cargo test --test cli`

Expected: compilation or test failure because the binary and parser do not exist yet.

- [ ] **Step 3: Implement beginner-readable argument parsing**

Walk the argument vector in flag/value pairs, store required values as `Option`s, apply only the specified `every=0` default, reject duplicates and unknown flags, validate all values, and return a short usage error through `main`.

- [ ] **Step 4: Connect `main` to `run` and stdout summaries**

Print a header followed by one tab-separated line for each temperature. Use `mean_abs_M` plus acceptance rate for Metropolis and mean cluster size for Wolff. Return a nonzero exit status for parse, simulation, or output errors.

- [ ] **Step 5: Run end-to-end tests and a small manual command**

Run: `cargo fmt --check && cargo test`

Then run:

```bash
cargo run -- --update metropolis --l 4 --t-from 1.5 --t-to 1.6 --t-step 0.05 --discard 2 --measure 3 --every 2 --seed 2026 --out runs/check
cargo run -- --update wolff --l 4 --t-from 1.5 --t-to 1.6 --t-step 0.05 --discard 2 --measure 3 --every 2 --seed 2026 --out runs/check-wolff
```

Expected: both commands succeed, produce valid JSON/JSONL files, and print four stdout lines including the header.

### Task 5: Review, final validation, and commit

**Files:**
- Review: all files created under `week3/`

- [ ] **Step 1: Run the focused final checks**

Run: `cargo fmt --check && cargo test`

This detects formatting errors, unit/integration regressions, parser failures, and malformed output behavior covered by the tests. If a check fails, fix the implementation and rerun the affected test once before repeating the full suite.

- [ ] **Step 2: Request read-only code review**

Review the final commit range against `ising.design.toml`, focusing on periodic-neighbor multiplicity, with-replacement Metropolis proposals, Wolff cluster membership, temperature continuation, sweep numbering, six-decimal JSONL formatting, and the no-unstated-defaults CLI contract.

- [ ] **Step 3: Inspect only the exact staged paths and staged diff**

Stage `Cargo.toml`, `Cargo.lock`, `src/lib.rs`, `src/main.rs`, `tests/lattice.rs`, `tests/updates.rs`, `tests/ramp.rs`, `tests/cli.rs`, and the approved plan only if it remains part of the intended deliverable. Confirm no unrelated parent artifacts are staged.

- [ ] **Step 4: Commit the focused Week 3 implementation**

```bash
git commit -m "feat: add seeded 2D Ising sampler"
```

Report the commit ID, tests run, and any validation limits without claiming scientific calibration beyond the implemented finite-size sampler.
