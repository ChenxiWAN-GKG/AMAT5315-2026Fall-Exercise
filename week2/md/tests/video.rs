use std::fs::{self, File};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use md::io::{RunMetadata, TrajectoryFrame, write_run_metadata, write_trajectory_frame};
use md::rdf::RdfBin;
use md::video::{render_rgb_frame, render_video};

fn metadata() -> RunMetadata {
    RunMetadata {
        n: 2,
        rho: 0.1,
        box_size: [10.0, 10.0],
        dt: 0.01,
        temperature: 0.5,
        eq_steps: 0,
        steps: 3,
        sample_every: 1,
        seed: 2026,
        integrator: "velocity-verlet".to_string(),
    }
}

fn frame(step: usize, x0: f64, x1: f64) -> TrajectoryFrame {
    TrajectoryFrame {
        step,
        t: step as f64 * 0.01,
        pos: vec![[x0, 2.0], [x1, 2.0]],
        vel: vec![[0.0, 0.0], [0.0, 0.0]],
        e_pot: 0.0,
        e_kin: 0.0,
    }
}

fn unique_output_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("md-video-{label}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[test]
fn renderer_returns_one_rgb_byte_triplet_per_pixel() {
    let frame = frame(1, 1.0, 3.0);
    let rdf = vec![RdfBin { r: 0.5, g: 1.0 }];
    let width = 64;
    let height = 36;

    let rgb = render_rgb_frame(&metadata(), &frame, &rdf, width, height);

    assert_eq!(rgb.len(), width * height * 3);
}

#[test]
fn video_writes_one_compact_movie_for_saved_frames() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg executable unavailable; video integration test not run");
        return;
    }

    let output_dir = unique_output_dir("integration");
    let run_metadata = metadata();
    write_run_metadata(&output_dir, &run_metadata).unwrap();
    let frames = [
        frame(1, 1.0, 3.0),
        frame(2, 1.1, 3.1),
        frame(3, 1.2, 3.2),
    ];
    let mut trajectory = File::create(output_dir.join("traj.jsonl")).unwrap();
    for saved_frame in &frames {
        write_trajectory_frame(&mut trajectory, saved_frame).unwrap();
    }

    let output = output_dir.join("run.mp4");
    render_video(&output_dir, &output).unwrap();

    assert!(output.is_file());
    let size = fs::metadata(output).unwrap().len();
    assert!(size > 0);
    assert!(size < 2_000_000);
}

#[test]
fn video_cli_reports_clear_encoder_error_when_ffmpeg_is_missing() {
    if ffmpeg_available() {
        eprintln!("SKIP: ffmpeg is available; missing-encoder path not exercised");
        return;
    }

    let output_dir = unique_output_dir("missing-encoder");
    let run_metadata = metadata();
    write_run_metadata(&output_dir, &run_metadata).unwrap();
    let mut trajectory = File::create(output_dir.join("traj.jsonl")).unwrap();
    write_trajectory_frame(&mut trajectory, &frame(1, 1.0, 3.0)).unwrap();
    let output = output_dir.join("run.mp4");

    let result = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["video"])
        .arg(&output_dir)
        .args(["--out"])
        .arg(&output)
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("ffmpeg encoder unavailable"), "{stderr}");
}
