use std::fs;
use std::process::Command;

use ndarray::{Array3, Ix3};
use tempfile::tempdir;

#[test]
fn forward_command_writes_traces_metadata_and_recording() {
    let directory = tempdir().unwrap();
    let experiment_path = directory.path().join("experiment.json");
    let output_path = directory.path().join("output");
    let background = "[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]";
    let zeros = "[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]";
    let json = format!(
        r#"{{
            "nx": 5, "nz": 5, "dx": 1.0, "dt": 0.2, "steps": 2,
            "source_frequency": 0.08, "source_peak_time": 1.5,
            "source_amplitude": 1.0, "shots": [[2,2]],
            "receivers": [[2,2]], "sponge_width": 1,
            "sponge_strength": 0.4, "background": {background},
            "perturbation": {zeros}, "length_unit_m": 100.0,
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
            "forward",
            "--every",
            "1",
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

    let traces: ndarray::Array3<f64> =
        ndarray_npy::read_npy(output_path.join("traces.npy")).unwrap();
    assert_eq!(traces.raw_dim(), Ix3(1, 2, 1));
    assert!(output_path.join("run.json").exists());
    assert!(output_path.join("result.json").exists());
    let run: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output_path.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["recording"]["steps"], serde_json::json!([0, 1, 2]));
    assert_eq!(
        run["recording"]["times"],
        serde_json::json!([0.0, 0.2, 0.4])
    );

    let wavefield: Array3<f32> = ndarray_npy::read_npy(output_path.join("wavefield.npy")).unwrap();
    assert_eq!(wavefield.shape(), &[3, 5, 5]);
    for frame in wavefield.outer_iter() {
        assert!(frame.row(0).iter().all(|value| *value == 0.0));
        assert!(frame.row(4).iter().all(|value| *value == 0.0));
        assert!(frame.column(0).iter().all(|value| *value == 0.0));
        assert!(frame.column(4).iter().all(|value| *value == 0.0));
    }
    assert!(output_path.join("echo.npy").exists());
}
