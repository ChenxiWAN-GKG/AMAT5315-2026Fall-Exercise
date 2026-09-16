pub mod fluid;
pub mod io;
pub mod run;
pub mod check;
pub mod rdf;
pub mod video;

use std::error::Error;
use std::fmt;

/// A two-dimensional vector in reduced units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    fn subtract(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }

    fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// One atom's position and velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    pub position: Vec2,
    pub velocity: Vec2,
}

impl Atom {
    pub const fn new(position: Vec2, velocity: Vec2) -> Self {
        Self { position, velocity }
    }
}

/// The complete state of the two-atom system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoAtomState {
    pub atoms: [Atom; 2],
}

impl TwoAtomState {
    pub const fn new(first: Atom, second: Atom) -> Self {
        Self {
            atoms: [first, second],
        }
    }
}

/// One recorded point from an experiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergySample {
    pub step: usize,
    pub time: f64,
    pub total_energy: f64,
    pub energy_error: f64,
}

/// A state or time-step validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationError {
    InvalidTimeStep,
    OverlappingAtoms,
    NonFiniteState,
}

impl fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidTimeStep => "time step must be finite and positive",
            Self::OverlappingAtoms => "atoms must have a non-zero separation",
            Self::NonFiniteState => "simulation state or energy is not finite",
        };
        formatter.write_str(message)
    }
}

impl Error for SimulationError {}

/// A simulation failure annotated with its method and step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExperimentError {
    pub method: &'static str,
    pub step: usize,
    pub cause: SimulationError,
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} failed at step {}: {}",
            self.method, self.step, self.cause
        )
    }
}

impl Error for ExperimentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}

/// Return the confirmed two-atom initial state.
pub const fn initial_state() -> TwoAtomState {
    TwoAtomState::new(
        Atom::new(Vec2::new(-0.6, 0.0), Vec2::ZERO),
        Atom::new(Vec2::new(0.6, 0.0), Vec2::ZERO),
    )
}

fn validate_state(state: &TwoAtomState) -> Result<(), SimulationError> {
    if state
        .atoms
        .iter()
        .all(|atom| atom.position.is_finite() && atom.velocity.is_finite())
    {
        Ok(())
    } else {
        Err(SimulationError::NonFiniteState)
    }
}

/// Return the two accelerations from the Lennard-Jones pair force.
pub fn pair_accelerations(state: &TwoAtomState) -> Result<[Vec2; 2], SimulationError> {
    validate_state(state)?;
    let displacement = state.atoms[0].position.subtract(state.atoms[1].position);
    let distance = displacement.length();
    if !distance.is_finite() {
        return Err(SimulationError::NonFiniteState);
    }
    if !(distance > 0.0) {
        return Err(SimulationError::OverlappingAtoms);
    }

    let radial_force = lennard_jones_force(distance);
    let first = displacement.scale(radial_force / distance);
    let second = first.scale(-1.0);
    if first.is_finite() && second.is_finite() {
        Ok([first, second])
    } else {
        Err(SimulationError::NonFiniteState)
    }
}

/// Return kinetic plus pair potential energy for a two-atom state.
pub fn total_energy(state: &TwoAtomState) -> Result<f64, SimulationError> {
    validate_state(state)?;
    let displacement = state.atoms[0].position.subtract(state.atoms[1].position);
    let distance = displacement.length();
    if !distance.is_finite() {
        return Err(SimulationError::NonFiniteState);
    }
    if !(distance > 0.0) {
        return Err(SimulationError::OverlappingAtoms);
    }

    let kinetic_energy = state
        .atoms
        .iter()
        .map(|atom| 0.5 * (atom.velocity.x.powi(2) + atom.velocity.y.powi(2)))
        .sum::<f64>();
    let energy = kinetic_energy + lennard_jones_energy(distance);
    if energy.is_finite() {
        Ok(energy)
    } else {
        Err(SimulationError::NonFiniteState)
    }
}

/// The common interface for time-stepping algorithms.
pub trait Integrator {
    fn name(&self) -> &'static str;
    fn step(&self, state: &mut TwoAtomState, time_step: f64) -> Result<(), SimulationError>;
}

/// The explicit forward Euler integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardEuler;

impl Integrator for ForwardEuler {
    fn name(&self) -> &'static str {
        "euler"
    }

    fn step(&self, state: &mut TwoAtomState, time_step: f64) -> Result<(), SimulationError> {
        if !(time_step.is_finite() && time_step > 0.0) {
            return Err(SimulationError::InvalidTimeStep);
        }

        let accelerations = pair_accelerations(state)?;
        let old_velocities = [state.atoms[0].velocity, state.atoms[1].velocity];
        let mut next = *state;
        for index in 0..2 {
            next.atoms[index].position = next.atoms[index]
                .position
                .add(old_velocities[index].scale(time_step));
            next.atoms[index].velocity =
                old_velocities[index].add(accelerations[index].scale(time_step));
        }
        validate_state(&next)?;
        *state = next;
        Ok(())
    }
}

/// The velocity-Verlet integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn name(&self) -> &'static str {
        "verlet"
    }

    fn step(&self, state: &mut TwoAtomState, time_step: f64) -> Result<(), SimulationError> {
        if !(time_step.is_finite() && time_step > 0.0) {
            return Err(SimulationError::InvalidTimeStep);
        }

        let old_accelerations = pair_accelerations(state)?;
        let old_velocities = [state.atoms[0].velocity, state.atoms[1].velocity];
        let half_time_step_squared = 0.5 * time_step * time_step;
        let mut next = *state;
        for index in 0..2 {
            next.atoms[index].position = next.atoms[index]
                .position
                .add(old_velocities[index].scale(time_step))
                .add(old_accelerations[index].scale(half_time_step_squared));
        }

        let new_accelerations = pair_accelerations(&next)?;
        for index in 0..2 {
            let average_acceleration = old_accelerations[index]
                .add(new_accelerations[index])
                .scale(0.5);
            next.atoms[index].velocity =
                old_velocities[index].add(average_acceleration.scale(time_step));
        }
        validate_state(&next)?;
        *state = next;
        Ok(())
    }
}

/// Run one integrator from a fresh state and record every step, including step 0.
pub fn run_experiment<I: Integrator>(
    integrator: I,
    mut state: TwoAtomState,
    time_step: f64,
    steps: usize,
) -> Result<Vec<EnergySample>, ExperimentError> {
    if !(time_step.is_finite() && time_step > 0.0) {
        return Err(ExperimentError {
            method: integrator.name(),
            step: 0,
            cause: SimulationError::InvalidTimeStep,
        });
    }

    let reference_energy = lennard_jones_energy(1.2);
    let sample_energy = |state: &TwoAtomState, step: usize| {
        total_energy(state)
            .map(|energy| EnergySample {
                step,
                time: step as f64 * time_step,
                total_energy: energy,
                energy_error: energy - reference_energy,
            })
            .map_err(|cause| ExperimentError {
                method: integrator.name(),
                step,
                cause,
            })
    };

    let mut samples = Vec::with_capacity(steps + 1);
    samples.push(sample_energy(&state, 0)?);
    for step in 1..=steps {
        integrator
            .step(&mut state, time_step)
            .map_err(|cause| ExperimentError {
                method: integrator.name(),
                step,
                cause,
            })?;
        samples.push(sample_energy(&state, step)?);
    }
    Ok(samples)
}

/// Return the reduced Lennard-Jones pair energy at a given distance.
pub fn lennard_jones_energy(distance: f64) -> f64 {
    let inverse_distance = 1.0 / distance;
    let inverse_distance_to_sixth = inverse_distance.powi(6);

    4.0 * (inverse_distance_to_sixth.powi(2) - inverse_distance_to_sixth)
}

/// Return the signed reduced radial Lennard-Jones force at a given distance.
pub fn lennard_jones_force(distance: f64) -> f64 {
    let inverse_distance = 1.0 / distance;
    let inverse_distance_to_sixth = inverse_distance.powi(6);

    24.0 * inverse_distance * (2.0 * inverse_distance_to_sixth.powi(2) - inverse_distance_to_sixth)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn lennard_jones_pair_energy_at_distance_two() {
        let energy = lennard_jones_energy(2.0);
        let expected_energy = -63.0 / 1024.0;

        assert!((energy - expected_energy).abs() < TOLERANCE);
    }

    #[test]
    fn lennard_jones_pair_force_at_distance_two() {
        let force = lennard_jones_force(2.0);
        let expected_force = -93.0 / 512.0;

        assert!((force - expected_force).abs() < TOLERANCE);
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn initial_state_has_requested_geometry_and_zero_velocity() {
        let state = initial_state();

        assert_eq!(state.atoms[0].position, Vec2::new(-0.6, 0.0));
        assert_eq!(state.atoms[1].position, Vec2::new(0.6, 0.0));
        assert_eq!(state.atoms[0].velocity, Vec2::ZERO);
        assert_eq!(state.atoms[1].velocity, Vec2::ZERO);
    }

    #[test]
    fn pair_accelerations_are_equal_and_opposite() {
        let state = initial_state();
        let accelerations = pair_accelerations(&state).expect("initial state is valid");

        assert_close(accelerations[0].x, -accelerations[1].x);
        assert_close(accelerations[0].y, -accelerations[1].y);
    }

    #[test]
    fn forward_euler_advances_from_old_position_and_velocity() {
        let mut state = TwoAtomState::new(
            Atom::new(Vec2::new(-0.6, 0.0), Vec2::new(0.0, 1.0)),
            Atom::new(Vec2::new(0.6, 0.0), Vec2::new(0.0, -1.0)),
        );

        ForwardEuler.step(&mut state, 0.01).expect("valid step");

        assert_close(state.atoms[0].position.y, 0.01);
        assert_close(state.atoms[1].position.y, -0.01);
    }

    #[test]
    fn velocity_verlet_preserves_finite_state_for_one_step() {
        let mut state = initial_state();

        VelocityVerlet.step(&mut state, 0.01).expect("valid step");

        assert!(state.atoms.iter().all(|atom| {
            atom.position.x.is_finite()
                && atom.position.y.is_finite()
                && atom.velocity.x.is_finite()
                && atom.velocity.y.is_finite()
        }));
    }

    #[test]
    fn shared_runner_records_step_zero_and_requested_count() {
        let samples =
            run_experiment(ForwardEuler, initial_state(), 0.01, 3).expect("valid experiment");

        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].step, 0);
        assert_close(samples[3].time, 0.03);
        assert_close(samples[0].energy_error, 0.0);
    }

    #[test]
    fn runner_rejects_invalid_time_step_before_recording_step_zero() {
        let error = run_experiment(ForwardEuler, initial_state(), 0.0, 0)
            .expect_err("zero time step must be rejected");

        assert_eq!(error.method, "euler");
        assert_eq!(error.step, 0);
        assert_eq!(error.cause, SimulationError::InvalidTimeStep);
    }

    #[test]
    fn energy_rejects_non_finite_separation() {
        let state = TwoAtomState::new(
            Atom::new(Vec2::new(f64::MAX, 0.0), Vec2::ZERO),
            Atom::new(Vec2::new(-f64::MAX, 0.0), Vec2::ZERO),
        );

        assert_eq!(total_energy(&state), Err(SimulationError::NonFiniteState));
    }

    #[test]
    fn requested_energy_conservation_criteria_hold() {
        let euler =
            run_experiment(ForwardEuler, initial_state(), 0.01, 500).expect("Euler run is valid");
        let verlet = run_experiment(VelocityVerlet, initial_state(), 0.01, 5000)
            .expect("Verlet run is valid");
        let reference_energy = lennard_jones_energy(1.2).abs();
        let euler_final = euler[500].energy_error / reference_energy;
        let verlet_max = verlet
            .iter()
            .map(|sample| (sample.energy_error / reference_energy).abs())
            .fold(0.0, f64::max);

        assert!(euler_final.abs() > 0.5);
        assert!(verlet_max < 1e-3);
    }
}
