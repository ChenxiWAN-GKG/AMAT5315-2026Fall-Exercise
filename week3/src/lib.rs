use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Update {
    Metropolis,
    Wolff,
}

impl Update {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metropolis => "metropolis",
            Self::Wolff => "wolff",
        }
    }

    fn time_unit(self) -> &'static str {
        match self {
            Self::Metropolis => "sweep",
            Self::Wolff => "cluster_flip",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunConfig {
    pub update: Update,
    pub l: usize,
    pub t_from: f64,
    pub t_to: f64,
    pub t_step: f64,
    pub discard: usize,
    pub measure: usize,
    pub seed: u64,
    pub every: usize,
    pub out: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureSummary {
    pub temperature: f64,
    pub mean_abs_m: f64,
    pub reported_value: f64,
}

#[derive(Debug)]
pub enum SimulationError {
    InvalidConfig(String),
    Io(io::Error),
}

impl fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => formatter.write_str(message),
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
        }
    }
}

impl Error for SimulationError {}

impl From<io::Error> for SimulationError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lattice {
    side: usize,
    spins: Vec<i8>,
}

impl Lattice {
    pub fn all_up(side: usize) -> Self {
        assert!(side >= 2, "the lattice side must be at least 2");
        Self {
            side,
            spins: vec![1; side * side],
        }
    }

    pub fn magnetization(&self) -> f64 {
        let total = self.spins.iter().map(|&spin| spin as i32).sum::<i32>();
        total as f64 / self.spins.len() as f64
    }

    pub fn energy_per_site(&self) -> f64 {
        let interaction_sum = self
            .spins
            .iter()
            .enumerate()
            .map(|(index, &spin)| {
                spin as i32
                    * self
                        .neighbors(index)
                        .iter()
                        .map(|&neighbor| self.spins[neighbor] as i32)
                        .sum::<i32>()
            })
            .sum::<i32>();
        -(interaction_sum as f64) / (2.0 * self.spins.len() as f64)
    }

    pub fn delta_energy(&self, index: usize) -> i32 {
        let neighbor_sum = self
            .neighbors(index)
            .iter()
            .map(|&neighbor| self.spins[neighbor] as i32)
            .sum::<i32>();
        2 * self.spins[index] as i32 * neighbor_sum
    }

    pub fn flip(&mut self, index: usize) {
        self.spins[index] = -self.spins[index];
    }

    pub fn metropolis_step<R: Rng + ?Sized>(&mut self, temperature: f64, rng: &mut R) -> usize {
        let mut accepted = 0;
        for _ in 0..self.spins.len() {
            let index = rng.random_range(0..self.spins.len());
            let delta = self.delta_energy(index);
            let accept =
                delta <= 0 || rng.random_range(0.0..1.0) < (-(delta as f64) / temperature).exp();
            if accept {
                self.flip(index);
                accepted += 1;
            }
        }
        accepted
    }

    pub fn wolff_step<R: Rng + ?Sized>(&mut self, temperature: f64, rng: &mut R) -> usize {
        let seed = rng.random_range(0..self.spins.len());
        let target_spin = self.spins[seed];
        let bond_probability = 1.0 - (-2.0 / temperature).exp();
        let mut in_cluster = vec![false; self.spins.len()];
        let mut stack = vec![seed];
        in_cluster[seed] = true;

        while let Some(index) = stack.pop() {
            for neighbor in self.neighbors(index) {
                if !in_cluster[neighbor]
                    && self.spins[neighbor] == target_spin
                    && rng.random_range(0.0..1.0) < bond_probability
                {
                    in_cluster[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        let mut cluster_size = 0;
        for (index, &included) in in_cluster.iter().enumerate() {
            if included {
                self.flip(index);
                cluster_size += 1;
            }
        }
        cluster_size
    }

    fn neighbors(&self, index: usize) -> [usize; 4] {
        let row = index / self.side;
        let column = index % self.side;
        let up = ((row + self.side - 1) % self.side) * self.side + column;
        let down = ((row + 1) % self.side) * self.side + column;
        let left = row * self.side + (column + self.side - 1) % self.side;
        let right = row * self.side + (column + 1) % self.side;
        [up, down, left, right]
    }
}

pub fn run(config: &RunConfig) -> Result<Vec<TemperatureSummary>, SimulationError> {
    let temperatures = temperature_grid(config)?;
    let lattice_size = config
        .l
        .checked_mul(config.l)
        .ok_or_else(|| SimulationError::InvalidConfig("lattice is too large".to_string()))?;
    let total_steps = config
        .discard
        .checked_add(config.measure)
        .ok_or_else(|| SimulationError::InvalidConfig("step count is too large".to_string()))?;
    total_steps
        .checked_mul(lattice_size)
        .ok_or_else(|| SimulationError::InvalidConfig("proposal count is too large".to_string()))?;

    fs::create_dir_all(&config.out)?;
    let mut run_file = File::create(config.out.join("run.json"))?;
    write_run_json(&mut run_file, config, &temperatures)?;
    let mut series_file = File::create(config.out.join("series.jsonl"))?;
    let mut spins_file = if config.every > 0 {
        Some(File::create(config.out.join("spins.jsonl"))?)
    } else {
        None
    };

    let mut lattice = Lattice::all_up(config.l);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut cumulative_sweep = 0usize;
    let mut summaries = Vec::with_capacity(temperatures.len());

    for &temperature in &temperatures {
        let mut accepted = 0usize;
        let mut measured_abs_m = 0.0;
        let mut measured_cluster_size = 0usize;

        for _ in 0..config.discard {
            let step_result = perform_step(&mut lattice, config.update, temperature, &mut rng);
            accepted += step_result.accepted;
            cumulative_sweep = cumulative_sweep.checked_add(1).ok_or_else(|| {
                SimulationError::InvalidConfig("sweep count is too large".to_string())
            })?;
        }

        for sweep in 1..=config.measure {
            let step_result = perform_step(&mut lattice, config.update, temperature, &mut rng);
            accepted += step_result.accepted;
            measured_cluster_size += step_result.cluster_size;
            cumulative_sweep = cumulative_sweep.checked_add(1).ok_or_else(|| {
                SimulationError::InvalidConfig("sweep count is too large".to_string())
            })?;

            let magnetization = lattice.magnetization();
            let energy = lattice.energy_per_site();
            measured_abs_m += magnetization.abs();
            write_series_record(
                &mut series_file,
                config.update,
                config.l,
                temperature,
                sweep,
                magnetization,
                energy,
                step_result.cluster_size,
            )?;
            write_frame_if_due(
                spins_file.as_mut(),
                config.every,
                sweep,
                cumulative_sweep,
                config.l,
                temperature,
                &lattice,
            )?;
        }

        let mean_abs_m = measured_abs_m / config.measure as f64;
        let reported_value = match config.update {
            Update::Metropolis => accepted as f64 / (total_steps * lattice_size) as f64,
            Update::Wolff => measured_cluster_size as f64 / config.measure as f64,
        };
        summaries.push(TemperatureSummary {
            temperature,
            mean_abs_m,
            reported_value,
        });
    }

    Ok(summaries)
}

struct StepResult {
    accepted: usize,
    cluster_size: usize,
}

fn perform_step<R: Rng + ?Sized>(
    lattice: &mut Lattice,
    update: Update,
    temperature: f64,
    rng: &mut R,
) -> StepResult {
    match update {
        Update::Metropolis => StepResult {
            accepted: lattice.metropolis_step(temperature, rng),
            cluster_size: 0,
        },
        Update::Wolff => StepResult {
            accepted: 0,
            cluster_size: lattice.wolff_step(temperature, rng),
        },
    }
}

fn temperature_grid(config: &RunConfig) -> Result<Vec<f64>, SimulationError> {
    if config.l < 2 {
        return Err(SimulationError::InvalidConfig(
            "l must be at least 2".to_string(),
        ));
    }
    if !(config.t_from.is_finite() && config.t_from > 0.0) {
        return Err(SimulationError::InvalidConfig(
            "t-from must be finite and positive".to_string(),
        ));
    }
    if !(config.t_to.is_finite() && config.t_to >= config.t_from) {
        return Err(SimulationError::InvalidConfig(
            "t-to must be finite and at least t-from".to_string(),
        ));
    }
    if !(config.t_step.is_finite() && config.t_step > 0.0) {
        return Err(SimulationError::InvalidConfig(
            "t-step must be finite and positive".to_string(),
        ));
    }
    if config.measure == 0 {
        return Err(SimulationError::InvalidConfig(
            "measure must be positive".to_string(),
        ));
    }

    let mut temperatures = Vec::new();
    let mut index = 0usize;
    loop {
        let value = config.t_from + index as f64 * config.t_step;
        if !value.is_finite() {
            return Err(SimulationError::InvalidConfig(
                "temperature grid is too large".to_string(),
            ));
        }
        if value > config.t_to {
            if endpoint_is_reached(config, index) {
                temperatures.push(config.t_to);
            }
            break;
        }
        temperatures.push(value);
        index = index.checked_add(1).ok_or_else(|| {
            SimulationError::InvalidConfig("temperature grid is too large".to_string())
        })?;
    }
    Ok(temperatures)
}

fn endpoint_is_reached(config: &RunConfig, index: usize) -> bool {
    let ratio = (config.t_to - config.t_from) / config.t_step;
    let nearest_index = ratio.round();
    let tolerance = 2.0 * f64::EPSILON * ratio.abs().max(1.0);

    nearest_index >= 0.0
        && nearest_index == index as f64
        && (ratio - nearest_index).abs() <= tolerance
}

fn write_run_json(file: &mut File, config: &RunConfig, temperatures: &[f64]) -> io::Result<()> {
    write!(
        file,
        "{{\"L\":{},\"update\":\"{}\",\"t_grid\":[",
        config.l,
        config.update.as_str()
    )?;
    for (index, temperature) in temperatures.iter().enumerate() {
        if index > 0 {
            write!(file, ",")?;
        }
        write!(file, "{temperature:.6}")?;
    }
    writeln!(
        file,
        "],\"discard\":{},\"measure\":{},\"seed\":{},\"sample_every\":1,\"time_unit\":\"{}\"}}",
        config.discard,
        config.measure,
        config.seed,
        config.update.time_unit()
    )
}

fn write_series_record(
    file: &mut File,
    update: Update,
    side: usize,
    temperature: f64,
    sweep: usize,
    magnetization: f64,
    energy: f64,
    cluster_size: usize,
) -> io::Result<()> {
    match update {
        Update::Metropolis => writeln!(
            file,
            "{{\"L\":{},\"T\":{temperature:.6},\"sweep\":{},\"M\":{magnetization:.6},\"E\":{energy:.6}}}",
            side, sweep
        ),
        Update::Wolff => writeln!(
            file,
            "{{\"L\":{},\"T\":{temperature:.6},\"sweep\":{},\"M\":{magnetization:.6},\"E\":{energy:.6},\"cluster_size\":{}}}",
            side, sweep, cluster_size
        ),
    }
}

fn write_frame_if_due(
    file: Option<&mut File>,
    every: usize,
    measurement_sweep: usize,
    cumulative_sweep: usize,
    side: usize,
    temperature: f64,
    lattice: &Lattice,
) -> io::Result<()> {
    if every == 0 || measurement_sweep % every != 0 {
        return Ok(());
    }
    let Some(file) = file else {
        return Ok(());
    };
    write!(
        file,
        "{{\"L\":{},\"T\":{temperature:.6},\"sweep\":{},\"m\":{:.6},\"spins\":[",
        side,
        cumulative_sweep,
        lattice.magnetization()
    )?;
    for (index, spin) in lattice.spins.iter().enumerate() {
        if index > 0 {
            write!(file, ",")?;
        }
        write!(file, "{spin}")?;
    }
    writeln!(file, "]}}")
}
