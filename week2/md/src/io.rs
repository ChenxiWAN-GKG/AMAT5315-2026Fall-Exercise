use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetadata {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_size: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryFrame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

pub fn write_run_metadata(out: &Path, metadata: &RunMetadata) -> Result<()> {
    fs::create_dir_all(out)?;
    let mut file = File::create(out.join("run.json"))?;
    serde_json::to_writer_pretty(&mut file, metadata)?;
    file.write_all(b"\n")?;
    Ok(())
}

pub fn write_trajectory_frame(
    writer: &mut impl Write,
    frame: &TrajectoryFrame,
) -> Result<()> {
    serde_json::to_writer(&mut *writer, frame)?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub fn read_run_metadata(out: &Path) -> Result<RunMetadata> {
    let file = File::open(out.join("run.json"))?;
    Ok(serde_json::from_reader(file)?)
}

pub fn read_trajectory(out: &Path) -> Result<Vec<TrajectoryFrame>> {
    let file = File::open(out.join("traj.jsonl"))?;
    let mut frames = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        frames.push(serde_json::from_str(&line)?);
    }
    Ok(frames)
}
