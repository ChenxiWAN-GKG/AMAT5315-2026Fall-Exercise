use std::path::PathBuf;
use std::process::Command;

fn unique_test_directory(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("ising-cli-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn cli_runs_both_updates_and_emits_one_summary_line_per_temperature() {
    for update in ["metropolis", "wolff"] {
        let output_path = unique_test_directory(update);
        let output = output_path.to_str().unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_ising"))
            .args([
                "--update",
                update,
                "--l",
                "2",
                "--t-from",
                "1.5",
                "--t-to",
                "1.6",
                "--t-step",
                "0.05",
                "--discard",
                "1",
                "--measure",
                "2",
                "--seed",
                "2026",
                "--out",
                output,
            ])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(String::from_utf8_lossy(&result.stdout).lines().count(), 4);
        assert!(!output_path.join("spins.jsonl").exists());
    }
}

#[test]
fn cli_rejects_zero_temperature_and_missing_required_flags() {
    let zero = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(["--update", "metropolis", "--l", "2", "--t-from", "0"])
        .output()
        .unwrap();
    assert!(!zero.status.success());

    let missing = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(["--update", "metropolis"])
        .output()
        .unwrap();
    assert!(!missing.status.success());
}
