use md::io::{RunMetadata, TrajectoryFrame};
use md::rdf::rdf_for_frames;

fn metadata() -> RunMetadata {
    RunMetadata {
        n: 2,
        rho: 1.0 / std::f64::consts::PI,
        box_size: [10.0, 10.0],
        dt: 0.01,
        temperature: 0.5,
        eq_steps: 0,
        steps: 2,
        sample_every: 1,
        seed: 2026,
        integrator: "velocity-verlet".to_string(),
        ramp_to: None,
    }
}

fn frame(x0: f64, x1: f64) -> TrajectoryFrame {
    TrajectoryFrame {
        step: 1,
        t: 0.01,
        pos: vec![[x0, 5.0], [x1, 5.0]],
        vel: vec![[0.0, 0.0], [0.0, 0.0]],
        e_pot: 0.0,
        e_kin: 0.0,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1.0e-12, "{actual} != {expected}");
}

#[test]
fn rdf_uses_minimum_image_distance_and_both_neighbour_directions() {
    let bins = rdf_for_frames(&metadata(), &[frame(0.1, 9.9)], 0, 5);

    assert_eq!(bins.len(), 5);
    assert_close(bins[0].r, 0.5);
    assert_close(bins[0].g, 1.0);
    for bin in &bins[1..] {
        assert_close(bin.g, 0.0);
    }
}

#[test]
fn rdf_averages_over_atoms_and_frames_with_annular_normalization() {
    let frames = [frame(0.1, 9.9), frame(1.0, 2.2)];
    let bins = rdf_for_frames(&metadata(), &frames, 1, 5);

    // Each pair contributes two directions.  With two atoms and two frames,
    // each occupied bin has measured count 2 / (2 * 2) = 0.5 per atom/frame.
    // The annular expectations are rho*pi*1 = 1 and rho*pi*3 = 3.
    assert_close(bins[0].g, 0.5);
    assert_close(bins[1].g, 1.0 / 6.0);
}

#[test]
fn rdf_through_frame_is_cumulative_and_inclusive() {
    let frames = [frame(0.1, 9.9), frame(1.0, 2.2)];
    let first = rdf_for_frames(&metadata(), &frames, 0, 5);
    let through_second = rdf_for_frames(&metadata(), &frames, 1, 5);

    assert_close(first[0].g, 1.0);
    assert_close(first[1].g, 0.0);
    assert_close(through_second[0].g, 0.5);
    assert_close(through_second[1].g, 1.0 / 6.0);
}
