use std::path::{Path, PathBuf};

use clap::Parser;
use ndarray::{Array2, Array3, ArrayView2, s};
use ndarray_npy::{read_npy, write_npy};
use serde_json::{Value, json};

use seismic::experiment::{Experiment, load_experiment};
use seismic::solver::{
    advance, build_sponge, enzyme_step_cell, enzyme_step_cell_reverse, gaussian_footprint, ricker,
    sample_receivers,
};

#[derive(Debug, Parser)]
#[command(about = "Simulate the Week 5 acoustic seismic experiment")]
struct Args {
    #[arg(long)]
    experiment: PathBuf,
    #[arg(long)]
    mode: String,
    #[arg(long)]
    every: Option<usize>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    data: Option<PathBuf>,
}

struct ShotResult {
    traces: Array2<f64>,
    frames: Vec<Array2<f32>>,
}

struct AdjointShotResult {
    image: Array2<f64>,
    frames: Vec<Array2<f32>>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    if args.every == Some(0) {
        return Err("--every must be positive".to_string());
    }
    if args.mode != "forward" && args.mode != "born" && args.mode != "adjoint" {
        return Err(format!(
            "unsupported mode '{}'; only forward, born, and adjoint are available",
            args.mode
        ));
    }
    let experiment = load_experiment(&args.experiment)?;
    std::fs::create_dir_all(&args.out)
        .map_err(|error| format!("create output directory: {error}"))?;
    if args.mode == "born" {
        return run_born(&args, &experiment);
    }
    if args.mode == "adjoint" {
        return run_adjoint(&args, &experiment);
    }
    write_metadata(&args, &experiment, "forward")?;

    let sigma = build_sponge(
        experiment.nx,
        experiment.nz,
        experiment.sponge_width as f64,
        experiment.sponge_strength,
    );
    let mut traces = Array3::zeros((
        experiment.shots.len(),
        experiment.steps,
        experiment.receivers.len(),
    ));
    let mut first_background_frames = Vec::new();

    for (shot_index, &shot) in experiment.shots.iter().enumerate() {
        let result = run_shot(
            &experiment,
            &experiment.background,
            &sigma,
            shot,
            args.every,
        );
        traces
            .slice_mut(s![shot_index, .., ..])
            .assign(&result.traces);
        if shot_index == 0 {
            first_background_frames = result.frames;
        }
    }
    write_npy(args.out.join("traces.npy"), &traces)
        .map_err(|error| format!("write traces.npy: {error}"))?;

    if let Some(every) = args.every {
        let perturbed_speed = &experiment.background + &experiment.perturbation;
        let shot = experiment.shots[0];
        let perturbed = run_shot(&experiment, &perturbed_speed, &sigma, shot, Some(every));
        let wavefield = stack_frames(&first_background_frames, experiment.nz, experiment.nx);
        let echo_frames = perturbed
            .frames
            .iter()
            .zip(first_background_frames.iter())
            .map(|(with_perturbation, background)| with_perturbation - background)
            .collect::<Vec<_>>();
        let echo = stack_frames(&echo_frames, experiment.nz, experiment.nx);
        write_npy(args.out.join("wavefield.npy"), &wavefield)
            .map_err(|error| format!("write wavefield.npy: {error}"))?;
        write_npy(args.out.join("echo.npy"), &echo)
            .map_err(|error| format!("write echo.npy: {error}"))?;
        update_recording_metadata(&args.out, every, experiment.steps, wavefield.shape()[0])?;
    }
    Ok(())
}

fn run_born(args: &Args, experiment: &Experiment) -> Result<(), String> {
    write_metadata(args, experiment, "born")?;
    let sigma = build_sponge(
        experiment.nx,
        experiment.nz,
        experiment.sponge_width as f64,
        experiment.sponge_strength,
    );
    let mut born_data = Array3::zeros((
        experiment.shots.len(),
        experiment.steps,
        experiment.receivers.len(),
    ));
    for (shot_index, &shot) in experiment.shots.iter().enumerate() {
        let traces = run_born_shot(experiment, &sigma, shot);
        born_data.slice_mut(s![shot_index, .., ..]).assign(&traces);
    }
    write_npy(args.out.join("born_data.npy"), &born_data)
        .map_err(|error| format!("write born_data.npy: {error}"))?;
    Ok(())
}

fn run_born_shot(experiment: &Experiment, sigma: &Array2<f64>, shot: [usize; 2]) -> Array2<f64> {
    let footprint = gaussian_footprint(experiment.nx, experiment.nz, shot[0], shot[1]);
    let mut previous = Array2::zeros((experiment.nz, experiment.nx));
    let mut current = Array2::zeros((experiment.nz, experiment.nx));
    let mut previous_dot = Array2::zeros((experiment.nz, experiment.nx));
    let mut current_dot = Array2::zeros((experiment.nz, experiment.nx));
    let mut next = Array2::zeros((experiment.nz, experiment.nx));
    let mut next_dot = Array2::zeros((experiment.nz, experiment.nx));
    let mut traces = Array2::zeros((experiment.steps, experiment.receivers.len()));

    for step in 0..experiment.steps {
        let source_value = experiment.source_amplitude
            * ricker(
                step as f64 * experiment.dt,
                experiment.source_frequency,
                experiment.source_peak_time,
            );
        let source = &footprint * source_value;
        next.fill(0.0);
        next_dot.fill(0.0);
        for z in 1..(experiment.nz - 1) {
            for x in 1..(experiment.nx - 1) {
                let (value, tangent) = enzyme_step_cell(
                    previous[[z, x]],
                    current[[z, x]],
                    current[[z, x - 1]],
                    current[[z, x + 1]],
                    current[[z - 1, x]],
                    current[[z + 1, x]],
                    experiment.background[[z, x]],
                    sigma[[z, x]],
                    source[[z, x]],
                    experiment.dx,
                    experiment.dt,
                    previous_dot[[z, x]],
                    current_dot[[z, x]],
                    current_dot[[z, x - 1]],
                    current_dot[[z, x + 1]],
                    current_dot[[z - 1, x]],
                    current_dot[[z + 1, x]],
                    experiment.perturbation[[z, x]],
                );
                next[[z, x]] = value;
                next_dot[[z, x]] = tangent;
            }
        }
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        std::mem::swap(&mut previous_dot, &mut current_dot);
        std::mem::swap(&mut current_dot, &mut next_dot);
        let samples = sample_receivers(&current_dot, &experiment.receivers);
        for (receiver, value) in samples.into_iter().enumerate() {
            traces[[step, receiver]] = value;
        }
    }
    traces
}

fn run_adjoint(args: &Args, experiment: &Experiment) -> Result<(), String> {
    let data_path = args
        .data
        .as_ref()
        .ok_or_else(|| "adjoint mode requires --data born_data.npy".to_string())?;
    let data: Array3<f64> = read_npy(data_path)
        .map_err(|error| format!("read adjoint data {}: {error}", data_path.display()))?;
    let expected_shape = [
        experiment.shots.len(),
        experiment.steps,
        experiment.receivers.len(),
    ];
    if data.shape() != expected_shape {
        return Err(format!(
            "adjoint data shape must be {:?}, got {:?}",
            expected_shape,
            data.shape()
        ));
    }
    write_metadata(args, experiment, "adjoint")?;
    let sigma = build_sponge(
        experiment.nx,
        experiment.nz,
        experiment.sponge_width as f64,
        experiment.sponge_strength,
    );
    let mut image = Array2::zeros((experiment.nz, experiment.nx));
    let mut first_adjoint_frames = Vec::new();
    for (shot_index, &shot) in experiment.shots.iter().enumerate() {
        let trajectory = forward_trajectory(experiment, &sigma, shot);
        let result = run_adjoint_shot(
            experiment,
            &sigma,
            &trajectory,
            shot,
            data.slice(s![shot_index, .., ..]),
            args.every,
        );
        image += &result.image;
        if shot_index == 0 {
            first_adjoint_frames = result.frames;
        }
    }
    write_npy(args.out.join("image.npy"), &image)
        .map_err(|error| format!("write image.npy: {error}"))?;
    if let Some(every) = args.every {
        let wavefield = stack_frames(&first_adjoint_frames, experiment.nz, experiment.nx);
        write_npy(args.out.join("wavefield.npy"), &wavefield)
            .map_err(|error| format!("write wavefield.npy: {error}"))?;
        update_adjoint_recording_metadata(
            &args.out,
            every,
            experiment.steps,
            wavefield.shape()[0],
        )?;
    }
    Ok(())
}

fn forward_trajectory(
    experiment: &Experiment,
    sigma: &Array2<f64>,
    shot: [usize; 2],
) -> Vec<(Array2<f64>, Array2<f64>)> {
    let footprint = gaussian_footprint(experiment.nx, experiment.nz, shot[0], shot[1]);
    let mut previous = Array2::zeros((experiment.nz, experiment.nx));
    let mut current = Array2::zeros((experiment.nz, experiment.nx));
    let mut next = Array2::zeros((experiment.nz, experiment.nx));
    let mut trajectory = vec![(previous.clone(), current.clone())];
    for step in 0..experiment.steps {
        let source_value = experiment.source_amplitude
            * ricker(
                step as f64 * experiment.dt,
                experiment.source_frequency,
                experiment.source_peak_time,
            );
        let source = &footprint * source_value;
        advance(
            &previous,
            &current,
            &mut next,
            &experiment.background,
            sigma,
            &source,
            experiment.dx,
            experiment.dt,
        );
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        trajectory.push((previous.clone(), current.clone()));
    }
    trajectory
}

fn run_adjoint_shot(
    experiment: &Experiment,
    sigma: &Array2<f64>,
    trajectory: &[(Array2<f64>, Array2<f64>)],
    shot: [usize; 2],
    data: ArrayView2<'_, f64>,
    every: Option<usize>,
) -> AdjointShotResult {
    let footprint = gaussian_footprint(experiment.nx, experiment.nz, shot[0], shot[1]);
    let mut output_bar = Array2::<f64>::zeros((experiment.nz, experiment.nx));
    let mut pending_bar = Array2::<f64>::zeros((experiment.nz, experiment.nx));
    let mut image = Array2::<f64>::zeros((experiment.nz, experiment.nx));
    let mut frames = Vec::new();
    if every.is_some() {
        frames.push(output_bar.mapv(|value| value as f32));
    }

    for step in (0..experiment.steps).rev() {
        for (receiver, &[x, z]) in experiment.receivers.iter().enumerate() {
            output_bar[[z, x]] += data[[step, receiver]];
        }
        let source_value = experiment.source_amplitude
            * ricker(
                step as f64 * experiment.dt,
                experiment.source_frequency,
                experiment.source_peak_time,
            );
        let source = &footprint * source_value;
        let (previous, current) = &trajectory[step];
        let mut previous_bar = Array2::<f64>::zeros((experiment.nz, experiment.nx));
        let mut current_bar = Array2::<f64>::zeros((experiment.nz, experiment.nx));
        for z in 1..(experiment.nz - 1) {
            for x in 1..(experiment.nx - 1) {
                let bars = enzyme_step_cell_reverse(
                    previous[[z, x]],
                    current[[z, x]],
                    current[[z, x - 1]],
                    current[[z, x + 1]],
                    current[[z - 1, x]],
                    current[[z + 1, x]],
                    experiment.background[[z, x]],
                    sigma[[z, x]],
                    source[[z, x]],
                    experiment.dx,
                    experiment.dt,
                    output_bar[[z, x]],
                );
                previous_bar[[z, x]] += bars[1];
                current_bar[[z, x]] += bars[2];
                current_bar[[z, x - 1]] += bars[3];
                current_bar[[z, x + 1]] += bars[4];
                current_bar[[z - 1, x]] += bars[5];
                current_bar[[z + 1, x]] += bars[6];
                image[[z, x]] += bars[7];
            }
        }
        output_bar = current_bar + &pending_bar;
        pending_bar = previous_bar;
        if every.is_some_and(|interval| step % interval == 0) || step == 0 {
            if every.is_some() {
                frames.push(output_bar.mapv(|value| value as f32));
            }
        }
    }
    AdjointShotResult { image, frames }
}

fn run_shot(
    experiment: &Experiment,
    speed: &Array2<f64>,
    sigma: &Array2<f64>,
    shot: [usize; 2],
    every: Option<usize>,
) -> ShotResult {
    let footprint = gaussian_footprint(experiment.nx, experiment.nz, shot[0], shot[1]);
    let mut previous = Array2::zeros((experiment.nz, experiment.nx));
    let mut current = Array2::zeros((experiment.nz, experiment.nx));
    let mut next = Array2::zeros((experiment.nz, experiment.nx));
    let mut traces = Array2::zeros((experiment.steps, experiment.receivers.len()));
    let mut frames = Vec::new();
    if every.is_some() {
        frames.push(current.mapv(|value| value as f32));
    }

    for step in 0..experiment.steps {
        let source_value = experiment.source_amplitude
            * ricker(
                step as f64 * experiment.dt,
                experiment.source_frequency,
                experiment.source_peak_time,
            );
        let source = &footprint * source_value;
        advance(
            &previous,
            &current,
            &mut next,
            speed,
            sigma,
            &source,
            experiment.dx,
            experiment.dt,
        );
        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
        let samples = sample_receivers(&current, &experiment.receivers);
        for (receiver, value) in samples.into_iter().enumerate() {
            traces[[step, receiver]] = value;
        }
        let completed_step = step + 1;
        if every.is_some_and(|interval| completed_step % interval == 0)
            || (every.is_some() && completed_step == experiment.steps)
        {
            frames.push(current.mapv(|value| value as f32));
        }
    }
    ShotResult { traces, frames }
}

fn stack_frames(frames: &[Array2<f32>], nz: usize, nx: usize) -> Array3<f32> {
    let mut output = Array3::zeros((frames.len(), nz, nx));
    for (index, frame) in frames.iter().enumerate() {
        output.slice_mut(s![index, .., ..]).assign(frame);
    }
    output
}

fn write_metadata(args: &Args, experiment: &Experiment, mode: &str) -> Result<(), String> {
    let run = json!({
        "experiment_file": args.experiment,
        "experiment": experiment_summary(experiment),
    });
    let result = json!({
        "mode": mode,
        "nx": experiment.nx,
        "nz": experiment.nz,
        "dx": experiment.dx,
        "dt": experiment.dt,
        "steps": experiment.steps,
        "shots": experiment.shots,
        "receivers": experiment.receivers,
    });
    write_json(&args.out.join("run.json"), &run)?;
    write_json(&args.out.join("result.json"), &result)
}

fn experiment_summary(experiment: &Experiment) -> Value {
    json!({
        "nx": experiment.nx,
        "nz": experiment.nz,
        "dx": experiment.dx,
        "dt": experiment.dt,
        "steps": experiment.steps,
        "source_frequency": experiment.source_frequency,
        "source_peak_time": experiment.source_peak_time,
        "source_amplitude": experiment.source_amplitude,
        "shots": experiment.shots,
        "receivers": experiment.receivers,
        "sponge_width": experiment.sponge_width,
        "sponge_strength": experiment.sponge_strength,
        "length_unit_m": experiment.length_unit_m,
        "time_unit_s": experiment.time_unit_s,
    })
}

fn update_recording_metadata(
    output: &Path,
    every: usize,
    steps: usize,
    frame_count: usize,
) -> Result<(), String> {
    let path = output.join("run.json");
    let mut run: Value = serde_json::from_reader(
        std::fs::File::open(&path).map_err(|error| format!("read run.json: {error}"))?,
    )
    .map_err(|error| format!("parse run.json: {error}"))?;
    run["recording"] = json!({
        "every": every,
        "steps": (0..frame_count)
            .map(|frame| if frame == frame_count - 1 { steps } else { frame * every })
            .collect::<Vec<_>>(),
        "times": (0..frame_count)
            .map(|frame| if frame == frame_count - 1 { steps } else { frame * every })
            .map(|step| step as f64)
            .collect::<Vec<_>>(),
    });
    write_json(&path, &run)
}

fn update_adjoint_recording_metadata(
    output: &Path,
    every: usize,
    steps: usize,
    frame_count: usize,
) -> Result<(), String> {
    let path = output.join("run.json");
    let mut run: Value = serde_json::from_reader(
        std::fs::File::open(&path).map_err(|error| format!("read run.json: {error}"))?,
    )
    .map_err(|error| format!("parse run.json: {error}"))?;
    let recorded_steps = (0..frame_count)
        .map(|frame| steps.saturating_sub(frame * every))
        .collect::<Vec<_>>();
    run["recording"] = json!({
        "every": every,
        "steps": recorded_steps,
        "times": (0..frame_count)
            .map(|frame| steps.saturating_sub(frame * every) as f64)
            .collect::<Vec<_>>(),
    });
    write_json(&path, &run)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(file, value)
        .map_err(|error| format!("write {}: {error}", path.display()))
}
