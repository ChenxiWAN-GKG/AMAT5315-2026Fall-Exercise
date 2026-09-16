# Two-Atom Molecular Dynamics Design

## Goal

Extend the `md` crate with a two-atom molecular dynamics experiment. The same
experiment runner must work with either a forward Euler integrator or a
velocity-Verlet integrator through one Rust trait. The experiment reports the
signed total-energy error over time as CSV, and a crate example plots the
relative error as `week2/dimer.png`.

## Scope

The model has two atoms in two spatial dimensions. Both atoms have mass
`m = 1`. They interact through the existing reduced Lennard-Jones potential and
move with open boundaries: positions are not wrapped or reflected.

The initial state is

- atom 1 position: `(-0.6, 0.0)`;
- atom 2 position: `(0.6, 0.0)`;
- both velocities: `(0.0, 0.0)`.

The initial separation is therefore `1.2`, the center of mass is at rest, and
the reference energy is `E0 = U(1.2)`.

For this two-atom experiment, this design does not add periodic boundaries,
more atoms, alternative potentials, command-line configuration, or external
dependencies beyond the macOS Quick Look utility already used by the field
renderer. The separate periodic-fluid CLI is documented in `README.md` and
does not change this dimer experiment contract.

## Data Model

The library will define a beginner-readable two-dimensional vector type with
the small set of operations needed by the simulation. An atom contains a
position vector and a velocity vector. A two-atom state contains exactly two
atoms.

The state will be copyable so that every integrator can begin with an identical
copy of the initial conditions.

## Lennard-Jones Acceleration

Let `d = r1 - r2` be the displacement from atom 2 to atom 1, and let
`r = |d|`. The existing function `lennard_jones_force(r)` returns the signed
radial force. The two accelerations are

```text
a1 = lennard_jones_force(r) d/r
a2 = -a1
```

Because each mass is 1, force and acceleration have the same numerical value.
A negative radial force attracts the atoms, while a positive radial force
repels them.

## Shared Integrator Interface

One trait will define the interface used by the experiment runner:

```rust
pub trait Integrator {
    fn name(&self) -> &'static str;
    fn step(
        &self,
        state: &mut TwoAtomState,
        time_step: f64,
    ) -> Result<(), SimulationError>;
}
```

The zero-sized types `ForwardEuler` and `VelocityVerlet` will both implement
this trait. The runner will accept any value implementing `Integrator`; it will
not contain method-specific stepping logic. Their `name` values will be
`euler` and `verlet`, respectively.

### Forward Euler

Forward Euler uses only positions, velocities, and accelerations from the
beginning of the step:

```text
r_i(new) = r_i(old) + v_i(old) dt
v_i(new) = v_i(old) + a_i(old) dt
```

Position and velocity updates must be computed before either old value is
overwritten.

### Velocity-Verlet

Velocity-Verlet first calculates the old acceleration, advances the positions,
recalculates acceleration at the new positions, and then advances velocities:

```text
r_i(new) = r_i(old) + v_i(old) dt + 0.5 a_i(old) dt^2
v_i(new) = v_i(old) + 0.5 [a_i(old) + a_i(new)] dt
```

## Experiment Runner

A shared `run_experiment` function will receive an integrator, a fresh copy of
the initial state, `dt`, and a step count. It will record step 0 before making
any update and then record one sample after each completed step.

The executable will run:

- forward Euler with `dt = 0.01` for 500 steps, producing samples through
  `t = 5`;
- velocity-Verlet with `dt = 0.01` for 5000 steps, producing samples through
  `t = 50`.

Thus the first 500 steps use identical initial conditions and run parameters,
while the Verlet series continues for the requested longer observation.

## Energy Measurement and CSV Output

At every sample, total energy is

```text
E(t) = 0.5 sum_i (v_ix^2 + v_iy^2) + U(r(t)).
```

The reported signed energy error is

```text
energy_error = E(t) - E0,
E0 = U(1.2).
```

For plotting, the signed relative error is

```text
relative_error = (E(t) - E0) / |E0|.
```

The executable will print one CSV stream to standard output with this header:

```text
method,step,time,total_energy,energy_error
```

Euler rows come first, followed by Verlet rows. There will be 501 Euler rows
and 5001 Verlet rows, excluding the header. Both step-0 rows must have zero
energy error. The integer step uses decimal notation, and every floating-point
column uses 15 digits after the decimal point in scientific notation.

The program will construct both complete result series before writing the CSV.
This prevents a simulation failure from leaving a partial output that looks
complete.

## Dimer Plot Example

The `dimer` example will run the shared experiment and save a two-panel PNG.
The left panel covers 500 steps (`0 <= t <= 5`) and plots both integrators'
signed relative errors. The right panel covers 5000 velocity-Verlet steps
(`0 <= t <= 50`) and plots `1000 * relative_error` for velocity-Verlet alone.

The selected color style uses orange for Euler, blue for velocity-Verlet, a
white background, a dashed zero line, equal-size panels, and one shared legend.
The right panel includes dashed reference lines at `+1` and `-1`, because
`|relative_error| < 10^-3` becomes `|1000 * relative_error| < 1`.

The example will be reproducible from the repository root with:

```sh
cargo run --manifest-path week2/md/Cargo.toml --example dimer -- week2/dimer.png
```

The example will generate an intermediate SVG in the system temporary
directory and convert it with `/usr/bin/qlmanage`, matching the existing field
renderer.

## Error Handling

The simulation will return a clear error for

- a time step that is non-finite or not positive;
- a separation that is zero or non-finite;
- non-finite position, velocity, acceleration, or energy values.

An experiment error will identify the integrator and step at which it occurred.
The executable will exit unsuccessfully rather than print invalid experimental
rows.

## File Responsibilities

- `src/lib.rs`: vector and state types, Lennard-Jones calculations, energy
  measurement, integrator trait and implementations, experiment runner, and
  focused unit tests. The original placeholder `greeting` function and its test
  will be removed because the simulation supersedes the hello-world behavior.
- `src/main.rs`: fixed experiment configuration and CSV output.
- `examples/dimer.rs`: fixed two-panel relative-energy-error plot and PNG
  conversion.
- `README.md`: reproduction command for `week2/dimer.png`.
- `src/bin/field.rs`: existing Lennard-Jones field visualization, unchanged
  except for any import adjustment required by the library API.

## Tests and Validation Boundary

Focused automated tests will verify

- the existing Lennard-Jones reference values;
- equal-and-opposite pair accelerations and their direction;
- one hand-calculated forward Euler step;
- one hand-calculated velocity-Verlet step;
- identical initial conditions and zero step-0 energy error for both methods;
- exact sample counts and final times;
- deterministic CSV header, method labels, and row ordering.
- the source-reported conservation criteria: velocity-Verlet maximum absolute
  relative error below `10^-3` over all 5000 steps, and forward Euler final
  absolute relative error above `0.5` at step 500;
- the dimer example's two panel labels, color assignments, and PNG output.

The conservation thresholds are **source-reported requirements**, not results
until the completed simulation is run. The example's figure is the visual
output for Chenxi to inspect independently.
