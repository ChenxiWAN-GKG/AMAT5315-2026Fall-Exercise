use std::io::Write;
use std::path::Path;

use seismic::experiment::load_experiment;
use tempfile::NamedTempFile;

fn write_experiment(json: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(json.as_bytes()).unwrap();
    file
}

fn valid_json() -> &'static str {
    r#"{
        "nx": 5, "nz": 4, "dx": 1.0, "dt": 0.2, "steps": 2,
        "source_frequency": 0.08, "source_peak_time": 1.5,
        "source_amplitude": 1.0,
        "shots": [[2, 1]], "receivers": [[1, 1], [3, 1]],
        "sponge_width": 1, "sponge_strength": 0.4,
        "background": [[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]],
        "perturbation": [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
        "length_unit_m": 100.0, "time_unit_s": 0.1
    }"#
}

#[test]
fn loads_a_valid_experiment() {
    let file = write_experiment(valid_json());
    let experiment = load_experiment(Path::new(file.path())).unwrap();
    assert_eq!(experiment.nx, 5);
    assert_eq!(experiment.nz, 4);
    assert_eq!(experiment.background.shape(), &[4, 5]);
    assert_eq!(experiment.shots, vec![[2, 1]]);
}

#[test]
fn rejects_a_background_shape_mismatch() {
    let json = valid_json().replace(
        "\"background\": [[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]]",
        "\"background\": [[1,1,1,1]]",
    );
    let file = write_experiment(&json);
    let error = load_experiment(Path::new(file.path())).unwrap_err();
    assert!(error.contains("background"));
}

#[test]
fn rejects_an_out_of_bounds_receiver() {
    let json = valid_json().replace("[3, 1]", "[5, 1]");
    let file = write_experiment(&json);
    let error = load_experiment(Path::new(file.path())).unwrap_err();
    assert!(error.contains("receiver"));
}
