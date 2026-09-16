use std::error::Error;
use std::io;
use std::path::Path;

use super::fluid::shifted_lj_potential;
use super::io::{RunMetadata, TrajectoryFrame, read_run_metadata, read_trajectory};

const RC: f64 = 2.5;
const TARGET_TEMPERATURE: f64 = 0.5;
const SPEED_BIN_COUNT: usize = 24;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, PartialEq)]
pub struct CheckReport {
    pub drift: f64,
    pub t_speed: f64,
    pub temperature_error: f64,
    pub chi_squared_22: f64,
    pub max_stored_energy_error: f64,
    pub drift_pass: bool,
    pub temperature_pass: bool,
    pub speed_shape_pass: bool,
}

pub fn check_artifacts(out: &Path) -> Result<CheckReport> {
    let metadata = read_run_metadata(out)?;
    let frames = read_trajectory(out)?;
    if frames.is_empty() {
        return Err(invalid_data("trajectory contains no frames"));
    }

    let mut totals = Vec::with_capacity(frames.len());
    let mut max_stored_energy_error: f64 = 0.0;
    for frame in &frames {
        let (e_pot, e_kin) = recompute_energy_components(&metadata, frame)?;
        let stored_potential_error = (frame.e_pot - e_pot).abs();
        let stored_kinetic_error = (frame.e_kin - e_kin).abs();
        if !(stored_potential_error.is_finite() && stored_kinetic_error.is_finite()) {
            return Err(invalid_data("stored energy is not finite"));
        }
        max_stored_energy_error = max_stored_energy_error
            .max(stored_potential_error)
            .max(stored_kinetic_error);
        let total = e_pot + e_kin;
        if !total.is_finite() {
            return Err(invalid_data("recomputed total energy is not finite"));
        }
        totals.push(total);
    }
    let k = usize::max(1, totals.len() / 10);
    let first_mean = totals[..k].iter().sum::<f64>() / k as f64;
    let last_mean = totals[totals.len() - k..].iter().sum::<f64>() / k as f64;
    let reference_energy = totals[0];
    if !(first_mean.is_finite()
        && last_mean.is_finite()
        && reference_energy.is_finite()
        && reference_energy != 0.0)
    {
        return Err(invalid_data("recomputed energies are not finite and non-zero"));
    }
    let drift = (last_mean - first_mean).abs() / reference_energy.abs();

    let mut pooled_v_squared = 0.0;
    let mut pooled_count = 0usize;
    let mut speeds = Vec::new();
    for frame in &frames {
        for velocity in &frame.vel {
            let speed_squared = velocity[0] * velocity[0] + velocity[1] * velocity[1];
            let speed = speed_squared.sqrt();
            if !(speed_squared.is_finite() && speed.is_finite()) {
                return Err(invalid_data("trajectory speed is not finite"));
            }
            pooled_v_squared += speed_squared;
            pooled_count += 1;
            speeds.push(speed);
        }
    }
    if pooled_count == 0 {
        return Err(invalid_data("trajectory contains no velocities"));
    }
    let t_speed = pooled_v_squared / (2.0 * pooled_count as f64);
    let temperature_error = (t_speed - TARGET_TEMPERATURE).abs();
    let mut observed = [0usize; SPEED_BIN_COUNT];
    for speed in speeds {
        observed[speed_bin(speed, t_speed)] += 1;
    }
    let expected = pooled_count as f64 / SPEED_BIN_COUNT as f64;
    let chi_squared_22 = observed
        .iter()
        .map(|count| (*count as f64 - expected).powi(2) / expected)
        .sum::<f64>()
        / 22.0;

    Ok(CheckReport {
        drift,
        t_speed,
        temperature_error,
        chi_squared_22,
        max_stored_energy_error,
        drift_pass: drift < 2.0e-3,
        temperature_pass: temperature_error < 0.05,
        speed_shape_pass: chi_squared_22 < 2.0,
    })
}

pub fn print_check_report(report: &CheckReport) {
    println!(
        "max_stored_energy_error = {:.8e} (diagnostic only)",
        report.max_stored_energy_error
    );
    println!(
        "drift = {:.8e} (limit < 2.0e-3): {}",
        report.drift,
        pass_fail(report.drift_pass)
    );
    println!(
        "T_speed = {:.8e} (target 0.5): {}",
        report.t_speed,
        pass_fail(report.temperature_pass)
    );
    println!(
        "temperature_error = {:.8e} (limit < 0.05): {}",
        report.temperature_error,
        pass_fail(report.temperature_pass)
    );
    println!(
        "chi_squared_22 = {:.8e} (limit < 2.0): {}",
        report.chi_squared_22,
        pass_fail(report.speed_shape_pass)
    );
}

fn pass_fail(passed: bool) -> &'static str {
    if passed { "PASS" } else { "FAIL" }
}

fn recompute_energy_components(
    metadata: &RunMetadata,
    frame: &TrajectoryFrame,
) -> Result<(f64, f64)> {
    if frame.pos.len() != metadata.n || frame.vel.len() != metadata.n {
        return Err(invalid_data("trajectory frame has the wrong particle count"));
    }
    let [lx, ly] = metadata.box_size;
    if !(lx.is_finite() && ly.is_finite() && lx > 0.0 && ly > 0.0) {
        return Err(invalid_data("metadata box lengths are invalid"));
    }
    if frame
        .pos
        .iter()
        .chain(frame.vel.iter())
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(invalid_data("trajectory contains a non-finite value"));
    }

    let mut e_pot = 0.0;
    for i in 0..frame.pos.len() {
        for j in (i + 1)..frame.pos.len() {
            let dx = minimum_image(frame.pos[i][0] - frame.pos[j][0], lx);
            let dy = minimum_image(frame.pos[i][1] - frame.pos[j][1], ly);
            let distance = dx.hypot(dy);
            if !distance.is_finite() {
                return Err(invalid_data("pair distance is not finite"));
            }
            if distance == 0.0 {
                return Err(invalid_data("trajectory contains overlapping atoms"));
            }
            e_pot += shifted_lj_potential(distance, RC);
        }
    }

    let e_kin = frame
        .vel
        .iter()
        .map(|velocity| 0.5 * (velocity[0] * velocity[0] + velocity[1] * velocity[1]))
        .sum::<f64>();
    if e_pot.is_finite() && e_kin.is_finite() {
        Ok((e_pot, e_kin))
    } else {
        Err(invalid_data("recomputed energy component is not finite"))
    }
}

fn minimum_image(delta: f64, length: f64) -> f64 {
    delta - length * (delta / length).round()
}

fn speed_bin(speed: f64, temperature: f64) -> usize {
    for upper_bin in 1..SPEED_BIN_COUNT {
        let probability = upper_bin as f64 / SPEED_BIN_COUNT as f64;
        let edge = (-2.0 * temperature * (1.0 - probability).ln()).sqrt();
        if speed < edge {
            return upper_bin - 1;
        }
    }
    SPEED_BIN_COUNT - 1
}

fn invalid_data(message: &str) -> Box<dyn Error> {
    Box::new(io::Error::new(
        io::ErrorKind::InvalidData,
        message.to_string(),
    ))
}
