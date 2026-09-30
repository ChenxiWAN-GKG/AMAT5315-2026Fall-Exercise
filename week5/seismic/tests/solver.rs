use ndarray::Array2;

use seismic::solver::{advance, build_sponge, gaussian_footprint, ricker, sample_receivers};

#[test]
fn sponge_is_strong_at_edge_and_zero_interior() {
    let sponge = build_sponge(5, 5, 2.0, 0.4);
    assert!((sponge[[2, 0]] - 0.4).abs() < 1e-12);
    assert!((sponge[[2, 1]] - 0.1).abs() < 1e-12);
    assert_eq!(sponge[[2, 2]], 0.0);
}

#[test]
fn source_functions_have_expected_peaks() {
    assert!((ricker(1.5, 0.08, 1.5) - 1.0).abs() < 1e-12);
    let footprint = gaussian_footprint(5, 5, 2, 3);
    assert!((footprint[[3, 2]] - 1.0).abs() < 1e-12);
    assert!((footprint[[3, 3]] - (-0.5_f64).exp()).abs() < 1e-12);
}

#[test]
fn one_step_update_uses_centered_laplacian_and_keeps_boundaries_zero() {
    let previous = Array2::zeros((5, 5));
    let mut current = Array2::zeros((5, 5));
    current[[2, 2]] = 1.0;
    let mut next = Array2::zeros((5, 5));
    let speed = Array2::from_elem((5, 5), 2.0);
    let sigma = Array2::zeros((5, 5));
    let mut source = Array2::zeros((5, 5));
    source[[2, 2]] = 3.0;

    advance(
        &previous, &current, &mut next, &speed, &sigma, &source, 1.0, 0.1,
    );

    assert!((next[[2, 2]] - 1.87).abs() < 1e-12);
    assert_eq!(next[[0, 0]], 0.0);
    assert_eq!(next[[0, 2]], 0.0);
    assert_eq!(next[[4, 4]], 0.0);
}

#[test]
fn receiver_coordinates_are_x_then_z() {
    let mut field = Array2::zeros((4, 5));
    field[[2, 1]] = 7.0;
    field[[1, 3]] = -2.0;
    let receivers = vec![[1, 2], [3, 1]];
    assert_eq!(sample_receivers(&field, &receivers), vec![7.0, -2.0]);
}
