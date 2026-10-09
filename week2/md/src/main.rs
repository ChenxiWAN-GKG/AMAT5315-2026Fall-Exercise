use md::EnergySample;
use md::run::{FluidConfig, run_simulation};
use std::fmt;
use std::path::PathBuf;
use std::process::ExitCode;
use md::fluid::ForceMethod;

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let command = match parse_args(&arguments) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let Command::Run { config, out } = command;
    match run_simulation(&config, &out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Command {
    Run {
        config: FluidConfig,
        out: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliError {
    message: String,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

fn parse_args(args: &[String]) -> Result<Command, CliError> {
    parse_run_args(args)
}

fn parse_run_args(args: &[String]) -> Result<Command, CliError> {
    let mut config = FluidConfig::contract_defaults();
    let mut out = None;
    let mut seen = 0_u16;
    let mut index = 0;

    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--n" => { config.n = parse_usize_value(args, &mut index, flag)?; seen |= 1; }
            "--rho" => { config.rho = parse_positive_float(args, &mut index, flag)?; seen |= 2; }
            "--temperature" => {
                config.temperature = parse_positive_float(args, &mut index, flag)?; seen |= 4;
            }
            "--dt" => { config.dt = parse_positive_float(args, &mut index, flag)?; seen |= 8; }
            "--eq-steps" => { config.eq_steps = parse_usize_value(args, &mut index, flag)?; seen |= 16; }
            "--steps" => { config.steps = parse_positive_usize(args, &mut index, flag)?; seen |= 32; }
            "--sample-every" => {
                config.sample_every = parse_positive_usize(args, &mut index, flag)?; seen |= 64;
            }
            "--seed" => { config.seed = parse_u64_value(args, &mut index, flag)?; seen |= 128; }
            "--out" => { out = Some(PathBuf::from(next_value(args, &mut index, flag)?)); seen |= 256; }
            "--force" => {
                config.force = match next_value(args, &mut index, flag)?.as_str() {
                    "naive" => ForceMethod::Naive,
                    "cells" => ForceMethod::Cells,
                    _ => return Err(CliError::usage("--force must be naive or cells")),
                }
            }
            "--ramp-to" => config.ramp_to = Some(parse_positive_float(args, &mut index, flag)?),
            _ => return Err(CliError::usage(format!("unknown flag: {flag}"))),
        }
        index += 1;
    }

    if seen != 511 {
        return Err(CliError::usage("required: --n --rho --temperature --dt --eq-steps --steps --sample-every --seed --out"));
    }
    if !is_supported_lattice_size(config.n) {
        return Err(CliError::usage("n must be a positive even square"));
    }
    Ok(Command::Run { config, out: out.expect("checked required arguments") })
}

fn next_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, CliError> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| CliError::usage(format!("missing value for {flag}")))
}

fn parse_usize_value(
    args: &[String],
    index: &mut usize,
    flag: &str,
) -> Result<usize, CliError> {
    next_value(args, index, flag)?.parse::<usize>().map_err(|_| {
        CliError::usage(format!("invalid integer for {flag}"))
    })
}

fn parse_positive_usize(
    args: &[String],
    index: &mut usize,
    flag: &str,
) -> Result<usize, CliError> {
    let value = parse_usize_value(args, index, flag)?;
    if value == 0 {
        return Err(CliError::usage(format!("{flag} must be positive")));
    }
    Ok(value)
}

fn parse_u64_value(
    args: &[String],
    index: &mut usize,
    flag: &str,
) -> Result<u64, CliError> {
    next_value(args, index, flag)?.parse::<u64>().map_err(|_| {
        CliError::usage(format!("invalid integer for {flag}"))
    })
}

fn parse_positive_float(
    args: &[String],
    index: &mut usize,
    flag: &str,
) -> Result<f64, CliError> {
    let value = next_value(args, index, flag)?
        .parse::<f64>()
        .map_err(|_| CliError::usage(format!("invalid number for {flag}")))?;
    if !(value.is_finite() && value > 0.0) {
        return Err(CliError::usage(format!("{flag} must be finite and positive")));
    }
    Ok(value)
}

fn is_supported_lattice_size(n: usize) -> bool {
    let side = (n as f64).sqrt() as usize;
    side > 0 && side % 2 == 0 && side.checked_mul(side) == Some(n)
}

#[allow(dead_code)]
fn format_sample(method: &str, sample: &EnergySample) -> String {
    format!(
        "{method},{},{:.15e},{:.15e},{:.15e}",
        sample.step, sample.time, sample.total_energy, sample.energy_error
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn formats_one_csv_sample_with_fixed_float_precision() {
        assert_eq!(
            super::format_sample(
                "euler",
                &EnergySample {
                    step: 0,
                    time: 0.0,
                    total_energy: -1.25,
                    energy_error: 0.0,
                },
            ),
            "euler,0,0.000000000000000e0,-1.250000000000000e0,0.000000000000000e0"
        );
    }

    #[test]
    fn parses_required_physics_and_output_path() {
        let command = parse_args(&arguments(&[
            "--n", "100", "--rho", "0.8", "--temperature", "0.5", "--dt", "0.01",
            "--eq-steps", "2000", "--steps", "10000", "--sample-every", "50",
            "--seed", "2026", "--out", "artifacts",
        ])).unwrap();

        match command {
            Command::Run { config, out } => {
                assert_eq!(config, md::run::FluidConfig::contract_defaults());
                assert_eq!(out, PathBuf::from("artifacts"));
            }
        }
    }

    #[test]
    fn rejects_unknown_run_flags() {
        let result = parse_args(&arguments(&["--unknown", "value"]));

        assert!(result.is_err());
    }

}
