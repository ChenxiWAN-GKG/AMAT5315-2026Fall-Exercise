use md::check::check_artifacts;
use md::run::{FluidConfig, run_simulation};

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("md-fluid-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn contract_run_passes_all_three_physics_bounds() {
    let out = unique_temp_dir("contract");
    run_simulation(&FluidConfig::contract_defaults(), &out).unwrap();
    let report = check_artifacts(&out).unwrap();
    assert!(report.drift < 2.0e-3);
    assert!((report.t_speed - 0.5).abs() < 0.05);
    assert!(report.chi_squared_22 < 2.0);
}
