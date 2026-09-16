use ising::{RunConfig, Update, run};
use std::path::{Path, PathBuf};

fn unique_test_directory(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ising-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn small_config(output: &Path, update: Update) -> RunConfig {
    RunConfig {
        update,
        l: 2,
        t_from: 1.5,
        t_to: 1.6,
        t_step: 0.05,
        discard: 1,
        measure: 2,
        seed: 2026,
        every: 0,
        out: output.to_path_buf(),
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn read_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn run_writes_a_temperature_grid_and_resets_measured_sweeps() {
    let output = unique_test_directory("ramp");
    let config = small_config(&output, Update::Metropolis);
    run(&config).unwrap();

    let run_json = read_json(&output.join("run.json"));
    assert_eq!(run_json["t_grid"], serde_json::json!([1.5, 1.55, 1.6]));
    assert_eq!(run_json["sample_every"], 1);
    let series = read_lines(&output.join("series.jsonl"));
    assert_eq!(series.len(), 6);
    assert!(series[0].contains("\"sweep\":1"));
    assert!(series[2].contains("\"sweep\":1"));
}

#[test]
fn spin_frames_use_the_cumulative_counter_and_six_decimal_numbers() {
    let output = unique_test_directory("frames");
    let mut config = small_config(&output, Update::Wolff);
    config.every = 2;
    run(&config).unwrap();

    let frames = read_lines(&output.join("spins.jsonl"));
    assert_eq!(frames.len(), 3);
    assert!(frames.iter().all(|line| line.contains(".000000")));
    assert!(frames[0].contains("\"sweep\":3"));
    assert!(frames[1].contains("\"sweep\":6"));
    assert!(frames[2].contains("\"sweep\":9"));
}

#[test]
fn temperature_grid_does_not_include_an_unreached_upper_bound() {
    let output = unique_test_directory("grid-endpoint");
    let mut config = small_config(&output, Update::Metropolis);
    config.t_from = 1.0;
    config.t_to = 1.0999999999999999;
    config.t_step = 0.1;
    config.discard = 0;
    run(&config).unwrap();

    let run_json = read_json(&output.join("run.json"));
    assert_eq!(run_json["t_grid"], serde_json::json!([1.0]));
}
