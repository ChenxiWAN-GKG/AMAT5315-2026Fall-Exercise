use std::process::Command;

#[test]
fn fluid_command_uses_flags_without_a_subcommand() {
    let out = std::env::temp_dir().join(format!("md-course-cli-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "--n", "100", "--rho", "0.8", "--temperature", "0.5", "--dt", "0.01",
            "--eq-steps", "0", "--steps", "1", "--sample-every", "1", "--seed", "2026",
            "--out", out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 2);
    assert_eq!(stdout.lines().next(), Some("t\tE_pot\tE_kin"));
}

#[test]
fn missing_physics_argument_is_a_usage_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["--n", "100"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}
