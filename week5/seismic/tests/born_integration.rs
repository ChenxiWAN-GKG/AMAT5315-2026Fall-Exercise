use std::fs;
use std::process::Command;

use ndarray_npy::read_npy;
use tempfile::tempdir;

#[test]
fn born_command_writes_velocity_perturbation_data() {
    let directory = tempdir().unwrap();
    let experiment_path = directory.path().join("experiment.json");
    let output_path = directory.path().join("output");
    let background = "[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]";
    let perturbation = "[[0,0,0,0,0],[0,0,0,0,0],[0,0,0.1,0,0],[0,0,0,0,0],[0,0,0,0,0]]";
    let json = format!(
        r#"{{
            "nx": 5, "nz": 5, "dx": 1.0, "dt": 0.2, "steps": 2,
            "source_frequency": 0.08, "source_peak_time": 1.5,
            "source_amplitude": 1.0, "shots": [[2,2]],
            "receivers": [[2,2]], "sponge_width": 1,
            "sponge_strength": 0.4, "background": {background},
            "perturbation": {perturbation}, "length_unit_m": 100.0,
            "time_unit_s": 0.1
        }}"#
    );
    fs::write(&experiment_path, json).unwrap();

    let binary = std::env::var("CARGO_BIN_EXE_seismic").expect("seismic binary path");
    let result = Command::new(binary)
        .args([
            "--experiment",
            experiment_path.to_str().unwrap(),
            "--mode",
            "born",
            "--out",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let born: ndarray::Array3<f64> = read_npy(output_path.join("born_data.npy")).unwrap();
    assert_eq!(born.shape(), &[1, 2, 1]);
    assert!(born.iter().any(|value| value.abs() > 0.0));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            &fs::read_to_string(output_path.join("result.json")).unwrap()
        )
        .unwrap()["mode"],
        "born"
    );
}
