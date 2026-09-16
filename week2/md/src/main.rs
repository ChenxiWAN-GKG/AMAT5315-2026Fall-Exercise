use md::EnergySample;
use md::run::{FluidConfig, run_simulation};
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match parse_args(&arguments)? {
        Command::Run { config, out } => run_simulation(&config, &out)?,
        Command::Check { out } => {
            let report = md::check::check_artifacts(&out)?;
            md::check::print_check_report(&report);
            if !(report.drift_pass && report.temperature_pass && report.speed_shape_pass) {
                return Err(Box::new(CliError::usage("checker acceptance failed")));
            }
        }
        Command::Video { input, output } => md::video::render_video(&input, &output)?,
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum Command {
    Run {
        config: FluidConfig,
        out: PathBuf,
    },
    Check {
        out: PathBuf,
    },
    Video {
        input: PathBuf,
        output: PathBuf,
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

impl Error for CliError {}

fn parse_args(args: &[String]) -> Result<Command, CliError> {
    match args.first().map(String::as_str) {
        Some("run") => parse_run_args(&args[1..]),
        Some("check") => parse_check_args(&args[1..]),
        Some("video") => parse_video_args(&args[1..]),
        _ => Err(CliError::usage("expected run, check, or video")),
    }
}

fn parse_run_args(args: &[String]) -> Result<Command, CliError> {
    let mut config = FluidConfig::contract_defaults();
    let mut out = PathBuf::from("artifacts");
    let mut index = 0;

    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--n" => config.n = parse_usize_value(args, &mut index, flag)?,
            "--rho" => config.rho = parse_positive_float(args, &mut index, flag)?,
            "--temperature" => {
                config.temperature = parse_positive_float(args, &mut index, flag)?
            }
            "--dt" => config.dt = parse_positive_float(args, &mut index, flag)?,
            "--eq-steps" => config.eq_steps = parse_usize_value(args, &mut index, flag)?,
            "--steps" => config.steps = parse_usize_value(args, &mut index, flag)?,
            "--sample-every" => {
                config.sample_every = parse_positive_usize(args, &mut index, flag)?
            }
            "--seed" => config.seed = parse_u64_value(args, &mut index, flag)?,
            "--out" => out = PathBuf::from(next_value(args, &mut index, flag)?),
            _ => return Err(CliError::usage(format!("unknown run flag: {flag}"))),
        }
        index += 1;
    }

    if !is_supported_lattice_size(config.n) {
        return Err(CliError::usage("n must be a positive even square"));
    }
    Ok(Command::Run { config, out })
}

fn parse_check_args(args: &[String]) -> Result<Command, CliError> {
    if args.len() != 1 {
        return Err(CliError::usage("check expects one artifact directory"));
    }
    Ok(Command::Check {
        out: PathBuf::from(&args[0]),
    })
}

fn parse_video_args(args: &[String]) -> Result<Command, CliError> {
    if args.len() != 3 || args[1] != "--out" {
        return Err(CliError::usage(
            "video expects an input directory and --out path",
        ));
    }
    Ok(Command::Video {
        input: PathBuf::from(&args[0]),
        output: PathBuf::from(&args[2]),
    })
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
    fn parses_run_defaults_and_output_path() {
        let command = parse_args(&arguments(&["run", "--out", "artifacts"])).unwrap();

        match command {
            Command::Run { config, out } => {
                assert_eq!(config, md::run::FluidConfig::contract_defaults());
                assert_eq!(out, PathBuf::from("artifacts"));
            }
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn rejects_unknown_run_flags() {
        let result = parse_args(&arguments(&["run", "--unknown", "value"]));

        assert!(result.is_err());
    }

    #[test]
    fn parses_check_and_video_paths() {
        let check = parse_args(&arguments(&["check", "artifacts"])).unwrap();
        match check {
            Command::Check { out } => assert_eq!(out, PathBuf::from("artifacts")),
            _ => panic!("expected check command"),
        }

        let video = parse_args(&arguments(&["video", "artifacts", "--out", "movie.mp4"])).unwrap();
        match video {
            Command::Video { input, output } => {
                assert_eq!(input, PathBuf::from("artifacts"));
                assert_eq!(output, PathBuf::from("movie.mp4"));
            }
            _ => panic!("expected video command"),
        }
    }
}
