use ising::{RunConfig, TemperatureSummary, Update, run};
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let config = parse_args(args)?;
    let summaries = run(&config)?;
    print_summary(config.update, &summaries);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CliError {}

fn parse_args(args: &[String]) -> Result<RunConfig, CliError> {
    let mut update = None;
    let mut l = None;
    let mut t_from = None;
    let mut t_to = None;
    let mut t_step = None;
    let mut discard = None;
    let mut measure = None;
    let mut seed = None;
    let mut every = 0;
    let mut every_seen = false;
    let mut out = None;
    let mut index = 0;

    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--update" => {
                ensure_absent(update, flag)?;
                let value = next_value(args, &mut index, flag)?;
                update = Some(match value {
                    "metropolis" => Update::Metropolis,
                    "wolff" => Update::Wolff,
                    _ => return Err(CliError(format!("unsupported update: {value}"))),
                });
            }
            "--l" => {
                ensure_absent(l, flag)?;
                l = Some(parse_usize(args, &mut index, flag)?);
            }
            "--t-from" => {
                ensure_absent(t_from, flag)?;
                t_from = Some(parse_positive_float(args, &mut index, flag)?);
            }
            "--t-to" => {
                ensure_absent(t_to, flag)?;
                t_to = Some(parse_positive_float(args, &mut index, flag)?);
            }
            "--t-step" => {
                ensure_absent(t_step, flag)?;
                t_step = Some(parse_positive_float(args, &mut index, flag)?);
            }
            "--discard" => {
                ensure_absent(discard, flag)?;
                discard = Some(parse_usize(args, &mut index, flag)?);
            }
            "--measure" => {
                ensure_absent(measure, flag)?;
                let value = parse_usize(args, &mut index, flag)?;
                if value == 0 {
                    return Err(CliError("--measure must be positive".to_string()));
                }
                measure = Some(value);
            }
            "--every" => {
                if every_seen {
                    return Err(CliError("duplicate flag: --every".to_string()));
                }
                every_seen = true;
                every = parse_usize(args, &mut index, flag)?;
            }
            "--seed" => {
                ensure_absent(seed, flag)?;
                seed = Some(parse_u64(args, &mut index, flag)?);
            }
            "--out" => {
                ensure_absent(out, flag)?;
                out = Some(PathBuf::from(next_value(args, &mut index, flag)?));
            }
            _ => return Err(CliError(format!("unknown flag: {flag}"))),
        }
        index += 1;
    }

    Ok(RunConfig {
        update: update.ok_or_else(|| CliError("missing required flag: --update".to_string()))?,
        l: l.ok_or_else(|| CliError("missing required flag: --l".to_string()))?,
        t_from: t_from.ok_or_else(|| CliError("missing required flag: --t-from".to_string()))?,
        t_to: t_to.ok_or_else(|| CliError("missing required flag: --t-to".to_string()))?,
        t_step: t_step.ok_or_else(|| CliError("missing required flag: --t-step".to_string()))?,
        discard: discard.ok_or_else(|| CliError("missing required flag: --discard".to_string()))?,
        measure: measure.ok_or_else(|| CliError("missing required flag: --measure".to_string()))?,
        seed: seed.ok_or_else(|| CliError("missing required flag: --seed".to_string()))?,
        every,
        out: out.ok_or_else(|| CliError("missing required flag: --out".to_string()))?,
    })
}

fn ensure_absent<T>(value: Option<T>, flag: &str) -> Result<(), CliError> {
    if value.is_some() {
        Err(CliError(format!("duplicate flag: {flag}")))
    } else {
        Ok(())
    }
}

fn next_value<'a>(args: &'a [String], index: &mut usize, flag: &str) -> Result<&'a str, CliError> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| CliError(format!("missing value for {flag}")))
}

fn parse_usize(args: &[String], index: &mut usize, flag: &str) -> Result<usize, CliError> {
    next_value(args, index, flag)?
        .parse::<usize>()
        .map_err(|_| CliError(format!("invalid integer for {flag}")))
}

fn parse_u64(args: &[String], index: &mut usize, flag: &str) -> Result<u64, CliError> {
    next_value(args, index, flag)?
        .parse::<u64>()
        .map_err(|_| CliError(format!("invalid integer for {flag}")))
}

fn parse_positive_float(args: &[String], index: &mut usize, flag: &str) -> Result<f64, CliError> {
    let value = next_value(args, index, flag)?
        .parse::<f64>()
        .map_err(|_| CliError(format!("invalid number for {flag}")))?;
    if !(value.is_finite() && value > 0.0) {
        return Err(CliError(format!("{flag} must be finite and positive")));
    }
    Ok(value)
}

fn print_summary(update: Update, summaries: &[TemperatureSummary]) {
    match update {
        Update::Metropolis => println!("T\tmean_abs_M\tacceptance_rate"),
        Update::Wolff => println!("T\tmean_abs_M\tmean_cluster_size"),
    }
    for summary in summaries {
        println!(
            "{:.6}\t{:.6}\t{:.6}",
            summary.temperature, summary.mean_abs_m, summary.reported_value
        );
    }
}
