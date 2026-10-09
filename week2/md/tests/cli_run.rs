use serde_json::Value;
use std::process::Command;

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("md-fluid-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn built_binary_writes_readable_run_and_trajectory_files() {
    let out = unique_temp_dir("cli");
    let status = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "--n",
            "16",
            "--rho",
            "0.8",
            "--temperature",
            "0.5",
            "--dt",
            "0.01",
            "--eq-steps",
            "50",
            "--steps",
            "20",
            "--sample-every",
            "10",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let metadata: Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["integrator"], "velocity-verlet");
    let lines = std::fs::read_to_string(out.join("traj.jsonl")).unwrap();
    let frames: Vec<Value> = lines
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0]["pos"].as_array().unwrap().len(), 16);
    assert_eq!(frames[0]["vel"].as_array().unwrap().len(), 16);
}
