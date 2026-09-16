use std::error::Error;
use std::fmt;

use super::Vec2;

/// A rectangular two-dimensional periodic simulation box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicBox {
    pub lx: f64,
    pub ly: f64,
}

impl PeriodicBox {
    pub const fn new(lx: f64, ly: f64) -> Self {
        Self { lx, ly }
    }

    /// Wrap a point into the half-open box intervals `[0, lx)` and `[0, ly)`.
    pub fn wrap(&self, point: Vec2) -> Vec2 {
        Vec2::new(
            wrap_coordinate(point.x, self.lx),
            wrap_coordinate(point.y, self.ly),
        )
    }

    /// Return the nearest-image displacement for a displacement vector.
    pub fn minimum_image(&self, displacement: Vec2) -> Vec2 {
        Vec2::new(
            minimum_image_component(displacement.x, self.lx),
            minimum_image_component(displacement.y, self.ly),
        )
    }
}

/// Positions and velocities for a periodic Lennard-Jones fluid.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidState {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
}

/// Errors raised while constructing or evaluating a fluid state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidError {
    InvalidConfiguration(&'static str),
    Overlap,
    NonFinite,
}

impl fmt::Display for FluidError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::Overlap => formatter.write_str("fluid state contains overlapping atoms"),
            Self::NonFinite => formatter.write_str("fluid state contains a non-finite value"),
        }
    }
}

impl Error for FluidError {}

/// Construct an even-sided staggered triangular lattice at the requested density.
pub fn triangular_lattice(
    n: usize,
    rho: f64,
) -> Result<(PeriodicBox, Vec<Vec2>), FluidError> {
    if !(rho.is_finite() && rho > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "density must be finite and positive",
        ));
    }

    let side = (n as f64).sqrt() as usize;
    if side == 0 || side.checked_mul(side) != Some(n) || side % 2 != 0 {
        return Err(FluidError::InvalidConfiguration(
            "n must be an even square",
        ));
    }

    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 3.0_f64.sqrt() * a / 2.0;
    let box_size = PeriodicBox::new(side as f64 * a, side as f64 * h);
    if !(a.is_finite()
        && h.is_finite()
        && box_size.lx.is_finite()
        && box_size.ly.is_finite()
        && box_size.lx > 0.0
        && box_size.ly > 0.0)
    {
        return Err(FluidError::NonFinite);
    }

    let mut positions = Vec::with_capacity(n);
    for row in 0..side {
        let shift = if row % 2 == 0 { 0.0 } else { 0.5 * a };
        for column in 0..side {
            let point = Vec2::new(column as f64 * a + shift, row as f64 * h);
            if !point.x.is_finite() || !point.y.is_finite() {
                return Err(FluidError::NonFinite);
            }
            positions.push(point);
        }
    }

    Ok((box_size, positions))
}

/// Return the Lennard-Jones potential shifted to zero at and beyond `rc`.
pub fn shifted_lj_potential(r: f64, rc: f64) -> f64 {
    if r >= rc {
        0.0
    } else {
        super::lennard_jones_energy(r) - super::lennard_jones_energy(rc)
    }
}

/// Accumulate pair forces using periodic minimum-image displacements.
pub fn total_forces(
    state: &FluidState,
    box_size: &PeriodicBox,
    rc: f64,
) -> Result<Vec<Vec2>, FluidError> {
    if !(box_size.lx.is_finite()
        && box_size.ly.is_finite()
        && box_size.lx > 0.0
        && box_size.ly > 0.0)
    {
        return Err(FluidError::InvalidConfiguration(
            "box lengths must be finite and positive",
        ));
    }
    if !(rc.is_finite() && rc > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "cutoff must be finite and positive",
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

    let mut forces = vec![Vec2::ZERO; state.positions.len()];
    for i in 0..state.positions.len() {
        for j in (i + 1)..state.positions.len() {
            let raw_displacement = Vec2::new(
                state.positions[i].x - state.positions[j].x,
                state.positions[i].y - state.positions[j].y,
            );
            let displacement = box_size.minimum_image(raw_displacement);
            if !displacement.x.is_finite() || !displacement.y.is_finite() {
                return Err(FluidError::NonFinite);
            }

            let distance = displacement.x.hypot(displacement.y);
            if !distance.is_finite() {
                return Err(FluidError::NonFinite);
            }
            if distance == 0.0 {
                return Err(FluidError::Overlap);
            }
            if distance >= rc {
                continue;
            }

            let magnitude = lj_force_magnitude(distance) / distance;
            let pair_force = Vec2::new(
                displacement.x * magnitude,
                displacement.y * magnitude,
            );
            if !pair_force.x.is_finite() || !pair_force.y.is_finite() {
                return Err(FluidError::NonFinite);
            }

            forces[i].x += pair_force.x;
            forces[i].y += pair_force.y;
            forces[j].x -= pair_force.x;
            forces[j].y -= pair_force.y;
            if !forces[i].x.is_finite()
                || !forces[i].y.is_finite()
                || !forces[j].x.is_finite()
                || !forces[j].y.is_finite()
            {
                return Err(FluidError::NonFinite);
            }
        }
    }

    Ok(forces)
}

/// Prepare deterministic Gaussian velocities with zero centre-of-mass motion.
pub fn prepare_velocities(
    n: usize,
    target_temperature: f64,
    seed: u64,
) -> Result<Vec<Vec2>, FluidError> {
    if n < 2 {
        return Err(FluidError::InvalidConfiguration(
            "at least two particles are required",
        ));
    }
    if !(target_temperature.is_finite() && target_temperature > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "temperature must be finite and positive",
        ));
    }

    let mut velocities = draw_gaussian_pairs(n, seed, target_temperature.sqrt());
    let mean_x = velocities.iter().map(|velocity| velocity.x).sum::<f64>() / n as f64;
    let mean_y = velocities.iter().map(|velocity| velocity.y).sum::<f64>() / n as f64;
    for velocity in &mut velocities {
        velocity.x -= mean_x;
        velocity.y -= mean_y;
    }
    rescale_velocities(&mut velocities, target_temperature)?;
    Ok(velocities)
}

/// Return the kinetic temperature after removing two centre-of-mass degrees of freedom.
pub fn thermostat_temperature(velocities: &[Vec2]) -> f64 {
    let degrees_of_freedom = 2.0 * velocities.len() as f64 - 2.0;
    if degrees_of_freedom <= 0.0 {
        return f64::NAN;
    }

    let kinetic_energy = velocities
        .iter()
        .map(|velocity| 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y))
        .sum::<f64>();
    2.0 * kinetic_energy / degrees_of_freedom
}

/// Rescale velocities so their thermostat temperature reaches the target value.
pub fn rescale_velocities(
    velocities: &mut [Vec2],
    target_temperature: f64,
) -> Result<(), FluidError> {
    if velocities.len() < 2 {
        return Err(FluidError::InvalidConfiguration(
            "at least two velocities are required",
        ));
    }
    if !(target_temperature.is_finite() && target_temperature > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "temperature must be finite and positive",
        ));
    }
    if velocities.iter().any(|velocity| !velocity.is_finite()) {
        return Err(FluidError::NonFinite);
    }

    let current_temperature = thermostat_temperature(velocities);
    if !(current_temperature.is_finite() && current_temperature > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "current temperature must be finite and positive",
        ));
    }

    let scale = (target_temperature / current_temperature).sqrt();
    if !scale.is_finite() {
        return Err(FluidError::NonFinite);
    }
    for velocity in velocities {
        velocity.x *= scale;
        velocity.y *= scale;
        if !velocity.is_finite() {
            return Err(FluidError::NonFinite);
        }
    }
    Ok(())
}

/// Advance a fluid state by one velocity-Verlet time step.
pub fn velocity_verlet_step(
    state: &mut FluidState,
    box_size: &PeriodicBox,
    dt: f64,
    rc: f64,
) -> Result<(), FluidError> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(FluidError::InvalidConfiguration(
            "time step must be finite and positive",
        ));
    }

    let old_forces = total_forces(state, box_size, rc)?;
    let half_dt_squared = 0.5 * dt * dt;
    let mut next_state = state.clone();
    for (index, position) in next_state.positions.iter_mut().enumerate() {
        position.x += state.velocities[index].x * dt + old_forces[index].x * half_dt_squared;
        position.y += state.velocities[index].y * dt + old_forces[index].y * half_dt_squared;
        *position = box_size.wrap(*position);
    }

    let new_forces = total_forces(&next_state, box_size, rc)?;
    for (index, velocity) in next_state.velocities.iter_mut().enumerate() {
        velocity.x += 0.5 * (old_forces[index].x + new_forces[index].x) * dt;
        velocity.y += 0.5 * (old_forces[index].y + new_forces[index].y) * dt;
        if !velocity.is_finite() {
            return Err(FluidError::NonFinite);
        }
    }

    if next_state
        .positions
        .iter()
        .chain(next_state.velocities.iter())
        .any(|vector| !vector.is_finite())
    {
        return Err(FluidError::NonFinite);
    }

    *state = next_state;
    Ok(())
}

fn draw_gaussian_pairs(n: usize, seed: u64, sigma: f64) -> Vec<Vec2> {
    let mut generator_state = seed;
    let mut velocities = Vec::with_capacity(n);
    for _ in 0..n {
        let u1 = uniform_open_interval(&mut generator_state);
        let u2 = uniform_open_interval(&mut generator_state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        velocities.push(Vec2::new(
            sigma * radius * angle.cos(),
            sigma * radius * angle.sin(),
        ));
    }
    velocities
}

fn uniform_open_interval(state: &mut u64) -> f64 {
    let value = splitmix64(state) >> 11;
    (value as f64 + 0.5) / 9_007_199_254_740_992.0
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn wrap_coordinate(value: f64, length: f64) -> f64 {
    value.rem_euclid(length)
}

fn minimum_image_component(delta: f64, length: f64) -> f64 {
    delta - length * (delta / length).round()
}

fn lj_force_magnitude(distance: f64) -> f64 {
    super::lennard_jones_force(distance)
}
