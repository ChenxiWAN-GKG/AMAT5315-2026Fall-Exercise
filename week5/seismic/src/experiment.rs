use std::fs::File;
use std::path::Path;

use ndarray::Array2;
use serde::Deserialize;

#[derive(Debug)]
pub struct Experiment {
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub shots: Vec<[usize; 2]>,
    pub receivers: Vec<[usize; 2]>,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub background: Array2<f64>,
    pub perturbation: Array2<f64>,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
}

#[derive(Debug, Deserialize)]
struct RawExperiment {
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    steps: usize,
    source_frequency: f64,
    source_peak_time: f64,
    source_amplitude: f64,
    shots: Vec<[f64; 2]>,
    receivers: Vec<[f64; 2]>,
    sponge_width: usize,
    sponge_strength: f64,
    background: Vec<Vec<f64>>,
    perturbation: Vec<Vec<f64>>,
    length_unit_m: f64,
    time_unit_s: f64,
}

pub fn load_experiment(path: &Path) -> Result<Experiment, String> {
    let file = File::open(path).map_err(|error| format!("open experiment: {error}"))?;
    let raw: RawExperiment = serde_json::from_reader(file)
        .map_err(|error| format!("parse experiment JSON: {error}"))?;
    let shots = convert_coordinates("shot", &raw.shots)?;
    let receivers = convert_coordinates("receiver", &raw.receivers)?;
    let background = make_array("background", raw.background, raw.nz, raw.nx)?;
    let perturbation = make_array("perturbation", raw.perturbation, raw.nz, raw.nx)?;
    let experiment = Experiment {
        nx: raw.nx,
        nz: raw.nz,
        dx: raw.dx,
        dt: raw.dt,
        steps: raw.steps,
        source_frequency: raw.source_frequency,
        source_peak_time: raw.source_peak_time,
        source_amplitude: raw.source_amplitude,
        shots,
        receivers,
        sponge_width: raw.sponge_width,
        sponge_strength: raw.sponge_strength,
        background,
        perturbation,
        length_unit_m: raw.length_unit_m,
        time_unit_s: raw.time_unit_s,
    };
    experiment.validate()?;
    Ok(experiment)
}

impl Experiment {
    pub fn validate(&self) -> Result<(), String> {
        if self.nx < 3 || self.nz < 3 {
            return Err("grid must have at least three cells in each direction".to_string());
        }
        if self.dx <= 0.0 || self.dt <= 0.0 {
            return Err("dx and dt must be positive".to_string());
        }
        if self.shots.is_empty() {
            return Err("shots must not be empty".to_string());
        }
        if self.receivers.is_empty() {
            return Err("receivers must not be empty".to_string());
        }
        if self.background.shape() != [self.nz, self.nx] {
            return Err(format!(
                "background shape must be [{}, {}]",
                self.nz, self.nx
            ));
        }
        if self.perturbation.shape() != [self.nz, self.nx] {
            return Err(format!(
                "perturbation shape must be [{}, {}]",
                self.nz, self.nx
            ));
        }
        for (index, &[x, z]) in self.shots.iter().enumerate() {
            if x >= self.nx || z >= self.nz {
                return Err(format!("shot {index} is out of bounds"));
            }
        }
        for (index, &[x, z]) in self.receivers.iter().enumerate() {
            if x >= self.nx || z >= self.nz {
                return Err(format!("receiver {index} is out of bounds"));
            }
        }
        Ok(())
    }
}

fn convert_coordinates(kind: &str, coordinates: &[[f64; 2]]) -> Result<Vec<[usize; 2]>, String> {
    coordinates
        .iter()
        .enumerate()
        .map(|(index, &[x, z])| {
            if !x.is_finite() || !z.is_finite() || x.fract() != 0.0 || z.fract() != 0.0 || x < 0.0 || z < 0.0 {
                return Err(format!("{kind} {index} coordinates must be nonnegative integers"));
            }
            Ok([x as usize, z as usize])
        })
        .collect()
}

fn make_array(
    name: &str,
    rows: Vec<Vec<f64>>,
    nz: usize,
    nx: usize,
) -> Result<Array2<f64>, String> {
    if rows.len() != nz || rows.iter().any(|row| row.len() != nx) {
        return Err(format!("{name} shape must be [{nz}, {nx}]"));
    }
    let values = rows.into_iter().flatten().collect::<Vec<_>>();
    Array2::from_shape_vec((nz, nx), values).map_err(|error| format!("{name} shape: {error}"))
}
