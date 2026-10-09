use std::fs;
use std::process::Command;

use ndarray_npy::read_npy;
use tempfile::tempdir;

#[test]
fn adjoint_command_reads_born_data_and_writes_image() {
    let directory = tempdir().unwrap();
    let experiment_path = directory.path().join("experiment.json");
    let born_path = directory.path().join("born");
    let adjoint_path = directory.path().join("adjoint");
    let background = "[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]";
    let perturbation = "[[0,0,0,0,0],[0,0,0,0,0],[0,0,0.1,0,0],[0,0,0,0,0],[0,0,0,0,0]]";
    let json = format!(
        r#"{{
            "nx": 5, "nz": 5, "dx": 1.0, "dt": 0.2, "steps": 4,
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

    let born = Command::new(&binary)
        .args([
            "--experiment",
            experiment_path.to_str().unwrap(),
            "--mode",
            "born",
            "--out",
            born_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        born.status.success(),
        "born stderr: {}",
        String::from_utf8_lossy(&born.stderr)
    );

    let adjoint = Command::new(&binary)
        .args([
            "--experiment",
            experiment_path.to_str().unwrap(),
            "--mode",
            "adjoint",
            "--data",
            born_path.join("born_data.npy").to_str().unwrap(),
            "--every",
            "1",
            "--out",
            adjoint_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        adjoint.status.success(),
        "adjoint stderr: {}",
        String::from_utf8_lossy(&adjoint.stderr)
    );

    let image: ndarray::Array2<f64> = read_npy(adjoint_path.join("image.npy")).unwrap();
    assert_eq!(image.shape(), &[5, 5]);
    assert!(image.iter().any(|value| value.abs() > 0.0));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            &fs::read_to_string(adjoint_path.join("result.json")).unwrap()
        )
        .unwrap()["mode"],
        "adjoint"
    );

    let checkpoint_path = directory.path().join("checkpoint");
    let checkpoint = Command::new(&binary)
        .args([
            "--experiment",
            experiment_path.to_str().unwrap(),
            "--mode",
            "adjoint",
            "--data",
            born_path.join("born_data.npy").to_str().unwrap(),
            "--storage",
            "treeverse",
            "--checkpoints",
            "2",
            "--out",
            checkpoint_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        checkpoint.status.success(),
        "checkpoint stderr: {}",
        String::from_utf8_lossy(&checkpoint.stderr)
    );
    let replayed: ndarray::Array2<f64> = read_npy(checkpoint_path.join("image.npy")).unwrap();
    let error = (&image - &replayed)
        .mapv(|value| value * value)
        .sum()
        .sqrt()
        / image.mapv(|value| value * value).sum().sqrt();
    assert!(error < 1e-12, "checkpoint image relative error: {error}");
    let actions: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(checkpoint_path.join("actions-0.json")).unwrap())
            .unwrap();
    let reverse_steps = actions
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["action"] == "grad")
        .map(|item| item["step"].as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(reverse_steps, vec![3, 2, 1, 0]);
    assert!(
        actions
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["saved_states"].as_u64().unwrap() <= 3)
    );
    let run: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(adjoint_path.join("run.json")).unwrap()).unwrap();
    assert_eq!(
        run["recording"]["steps"],
        serde_json::json!([4, 3, 2, 1, 0])
    );
    for (step, time) in run["recording"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .zip(run["recording"]["times"].as_array().unwrap())
    {
        assert!((time.as_f64().unwrap() - step.as_u64().unwrap() as f64 * 0.2).abs() < 1e-12);
    }
}
