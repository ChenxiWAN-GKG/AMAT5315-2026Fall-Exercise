use std::fs::{self, File};
use std::path::PathBuf;

use md::check::check_artifacts;
use md::io::{RunMetadata, TrajectoryFrame, write_run_metadata, write_trajectory_frame};

fn unique_output_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("md-check-{label}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn metadata() -> RunMetadata {
    RunMetadata {
        n: 2,
        rho: 0.1,
        box_size: [20.0, 20.0],
        dt: 0.01,
        temperature: 0.5,
        eq_steps: 0,
        steps: 20,
        sample_every: 1,
        seed: 2026,
        integrator: "velocity-verlet".to_string(),
    }
}

fn frame(step: usize, velocities: [[f64; 2]; 2], e_pot: f64, e_kin: f64) -> TrajectoryFrame {
    TrajectoryFrame {
        step,
        t: step as f64 * 0.01,
        pos: vec![[0.0, 0.0], [3.0, 0.0]],
        vel: velocities.to_vec(),
        e_pot,
        e_kin,
    }
}

fn write_artifacts(output: &PathBuf, frames: &[TrajectoryFrame]) {
    write_run_metadata(output, &metadata()).unwrap();
    let mut trajectory = File::create(output.join("traj.jsonl")).unwrap();
    for frame in frames {
        write_trajectory_frame(&mut trajectory, frame).unwrap();
    }
}

#[test]
fn checker_uses_two_frame_windows_and_recomputed_energies() {
    let output = unique_output_dir("windows");
    let energies = (0..20)
        .map(|step| match step {
            0 => 1.0_f64,
            1 => 3.0_f64,
            18 => 5.0_f64,
            19 => 7.0_f64,
            _ => 2.0_f64,
        })
        .collect::<Vec<_>>();
    let frames = energies
        .iter()
        .enumerate()
        .map(|(step, energy)| {
            frame(
                step + 1,
                [[(2.0 * energy).sqrt(), 0.0], [0.0, 0.0]],
                999.0,
                999.0,
            )
        })
        .collect::<Vec<_>>();
    write_artifacts(&output, &frames);

    let report = check_artifacts(&output).unwrap();

    // k = floor(20 / 10) = 2; E0 = 1, first mean = 2, last mean = 6.
    assert!((report.drift - 4.0).abs() < 1.0e-12);
    assert!(report.max_stored_energy_error > 0.0);
}

#[test]
fn checker_reports_speed_temperature_and_absolute_error() {
    let output = unique_output_dir("temperature");
    write_artifacts(
        &output,
        &[frame(1, [[2.0, 0.0], [2.0, 0.0]], -123.0, -456.0)],
    );

    let report = check_artifacts(&output).unwrap();

    assert!((report.t_speed - 2.0).abs() < 1.0e-12);
    assert!((report.temperature_error - 1.5).abs() < 1.0e-12);
}

#[test]
fn checker_uses_24_speed_bins_and_chi_squared_denominator_22() {
    let output = unique_output_dir("bins");
    let bin_indices = (0..24).map(|bin| if bin == 1 { 0 } else { bin });
    let speeds = bin_indices
        .map(|bin| {
            let probability = (bin as f64 + 0.5) / 24.0;
            (-2.0 * 0.5 * (1.0 - probability).ln()).sqrt()
        })
        .collect::<Vec<_>>();
    let frames = speeds
        .chunks_exact(2)
        .enumerate()
        .map(|(index, pair)| frame(index + 1, [[pair[0], 0.0], [pair[1], 0.0]], 0.0, 0.0))
        .collect::<Vec<_>>();
    write_artifacts(&output, &frames);

    let report = check_artifacts(&output).unwrap();

    // Counts are [2, 0, 1, ..., 1], expected M/24 = 1, then divide by 22.
    assert!((report.chi_squared_22 - 2.0 / 22.0).abs() < 1.0e-12);
}
