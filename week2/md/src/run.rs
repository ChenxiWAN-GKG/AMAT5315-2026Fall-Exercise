use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::fluid::{
    FluidError, FluidState, PeriodicBox, prepare_velocities, rescale_velocities,
    shifted_lj_potential, triangular_lattice, velocity_verlet_step,
};
use super::io::{RunMetadata, TrajectoryFrame, write_run_metadata, write_trajectory_frame};
use super::Vec2;

pub const RC: f64 = 2.5;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, PartialEq)]
pub struct FluidConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
}

impl FluidConfig {
    pub const fn contract_defaults() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2_000,
            steps: 10_000,
            sample_every: 50,
            seed: 2026,
        }
    }
}

pub fn run_simulation(config: &FluidConfig, out: &Path) -> Result<()> {
    validate_config(config)?;

    let (box_size, positions) = triangular_lattice(config.n, config.rho)?;
    let velocities = prepare_velocities(config.n, config.temperature, config.seed)?;
    let mut state = FluidState {
        positions,
        velocities,
    };

    let metadata = RunMetadata {
        n: config.n,
        rho: config.rho,
        box_size: [box_size.lx, box_size.ly],
        dt: config.dt,
        temperature: config.temperature,
        eq_steps: config.eq_steps,
        steps: config.steps,
        sample_every: config.sample_every,
        seed: config.seed,
        integrator: "velocity-verlet".to_string(),
    };
    write_run_metadata(out, &metadata)?;

    let trajectory_file = File::create(out.join("traj.jsonl"))?;
    let mut trajectory = BufWriter::new(trajectory_file);

    for eq_step in 1..=config.eq_steps {
        velocity_verlet_step(&mut state, &box_size, config.dt, RC)?;
        if eq_step % 50 == 0 {
            rescale_velocities(&mut state.velocities, config.temperature)?;
        }
    }

    for step in 1..=config.steps {
        velocity_verlet_step(&mut state, &box_size, config.dt, RC)?;
        if step % config.sample_every == 0 {
            let frame = frame_from_state(&state, step, config.dt, &box_size)?;
            write_trajectory_frame(&mut trajectory, &frame)?;
        }
    }
    trajectory.flush()?;
    Ok(())
}

fn validate_config(config: &FluidConfig) -> std::result::Result<(), FluidError> {
    if !(config.temperature.is_finite() && config.temperature > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "temperature must be finite and positive",
        ));
    }
    if !(config.dt.is_finite() && config.dt > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "time step must be finite and positive",
        ));
    }
    if config.sample_every == 0 {
        return Err(FluidError::InvalidConfiguration(
            "sample interval must be positive",
        ));
    }
    Ok(())
}

fn frame_from_state(
    state: &FluidState,
    step: usize,
    dt: f64,
    box_size: &PeriodicBox,
) -> std::result::Result<TrajectoryFrame, FluidError> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "time step must be finite and positive",
        ));
    }
    if !(box_size.lx.is_finite()
        && box_size.ly.is_finite()
        && box_size.lx > 0.0
        && box_size.ly > 0.0)
    {
        return Err(FluidError::InvalidConfiguration(
            "box lengths must be finite and positive",
        ));
    }
    if state.positions.len() != state.velocities.len() {
        return Err(FluidError::InvalidConfiguration(
            "positions and velocities must have equal lengths",
        ));
    }
    if state
        .positions
        .iter()
        .chain(state.velocities.iter())
        .any(|vector| !vector.x.is_finite() || !vector.y.is_finite())
    {
        return Err(FluidError::NonFinite);
    }

    let mut e_pot = 0.0;
    for i in 0..state.positions.len() {
        for j in (i + 1)..state.positions.len() {
            let displacement = box_size.minimum_image(Vec2::new(
                state.positions[i].x - state.positions[j].x,
                state.positions[i].y - state.positions[j].y,
            ));
            let distance = displacement.x.hypot(displacement.y);
            if !distance.is_finite() {
                return Err(FluidError::NonFinite);
            }
            if distance == 0.0 {
                return Err(FluidError::Overlap);
            }
            e_pot += shifted_lj_potential(distance, RC);
        }
    }

    let e_kin = state
        .velocities
        .iter()
        .map(|velocity| 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y))
        .sum::<f64>();
    let time = step as f64 * dt;
    if !(time.is_finite() && e_pot.is_finite() && e_kin.is_finite()) {
        return Err(FluidError::NonFinite);
    }

    Ok(TrajectoryFrame {
        step,
        t: time,
        pos: state
            .positions
            .iter()
            .map(|position| [position.x, position.y])
            .collect(),
        vel: state
            .velocities
            .iter()
            .map(|velocity| [velocity.x, velocity.y])
            .collect(),
        e_pot,
        e_kin,
    })
}
