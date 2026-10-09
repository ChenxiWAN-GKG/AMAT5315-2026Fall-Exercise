use std::collections::BTreeSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use md::io::{
    RunMetadata, TrajectoryFrame, read_run_metadata, read_trajectory, write_run_metadata,
    write_trajectory_frame,
};

fn unique_output_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("md-io-schema-{label}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn object_keys(value: &serde_json::Value) -> BTreeSet<String> {
    value
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

#[test]
fn run_metadata_has_exact_keys_and_round_trips() {
    let output = unique_output_dir("metadata");
    let metadata = RunMetadata {
        n: 16,
        rho: 0.8,
        box_size: [5.0, 4.0],
        dt: 0.01,
        temperature: 0.5,
        eq_steps: 50,
        steps: 20,
        sample_every: 10,
        seed: 2026,
        integrator: "velocity-verlet".to_string(),
        ramp_to: None,
    };

    write_run_metadata(&output, &metadata).unwrap();

    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output.join("run.json")).unwrap()).unwrap();
    let expected_keys = [
        "n",
        "rho",
        "box",
        "dt",
        "temperature",
        "eq_steps",
        "steps",
        "sample_every",
        "seed",
        "integrator",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<BTreeSet<_>>();

    assert_eq!(object_keys(&value), expected_keys);
    assert_eq!(value["integrator"], "velocity-verlet");
    assert_eq!(value["box"].as_array().unwrap().len(), 2);

    let round_trip = read_run_metadata(&output).unwrap();
    assert_eq!(round_trip.n, metadata.n);
    assert_eq!(round_trip.box_size, metadata.box_size);
    assert_eq!(round_trip.integrator, metadata.integrator);
}

#[test]
fn trajectory_jsonl_has_exact_keys_and_pair_arrays() {
    let output = unique_output_dir("trajectory");
    let frame = TrajectoryFrame {
        step: 10,
        t: 0.1,
        pos: vec![[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        vel: vec![[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
        e_pot: -1.25,
        e_kin: 0.75,
    };
    let second_frame = TrajectoryFrame {
        step: 20,
        t: 0.2,
        pos: vec![[4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0]],
        vel: vec![[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9]],
        e_pot: -1.0,
        e_kin: 0.5,
    };
    let mut trajectory = File::create(output.join("traj.jsonl")).unwrap();

    write_trajectory_frame(&mut trajectory, &frame).unwrap();
    write_trajectory_frame(&mut trajectory, &second_frame).unwrap();
    drop(trajectory);

    let lines = fs::read_to_string(output.join("traj.jsonl")).unwrap();
    assert!(lines.ends_with('\n'));
    let expected_keys = ["step", "t", "pos", "vel", "E_pot", "E_kin"]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    for line in lines.lines() {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(object_keys(&value), expected_keys);
        assert_eq!(value["pos"].as_array().unwrap().len(), 4);
        assert_eq!(value["vel"].as_array().unwrap().len(), 4);
        for pair in value["pos"].as_array().unwrap() {
            assert_eq!(pair.as_array().unwrap().len(), 2);
        }
        for pair in value["vel"].as_array().unwrap() {
            assert_eq!(pair.as_array().unwrap().len(), 2);
        }
    }

    let frames = read_trajectory(Path::new(&output)).unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].step, frame.step);
    assert_eq!(frames[1].e_kin, second_frame.e_kin);
}
