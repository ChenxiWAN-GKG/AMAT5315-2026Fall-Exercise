use std::f64::consts::PI;

use week4::{
    ExplicitMidpoint, ForwardEuler, Integrator, RungeKutta4,
    advection_diffusion_centered_rate, advection_diffusion_fourier_rate,
};

fn exact_wave(x: f64, t: f64, k: i32, c: f64, nu: f64) -> f64 {
    ((k as f64) * (x - c * t)).cos() * (-(nu * (k * k) as f64) * t).exp()
}

fn integrate<I: Integrator>(integrator: I, state: Vec<f64>, dt: f64, steps: usize) -> Vec<f64> {
    let rate = advection_diffusion_fourier_rate(64, 1.0, 0.05);
    let mut state = state;
    for _ in 0..steps {
        state = integrator.step(&state, dt, &rate);
    }
    state
}

#[test]
fn each_integrator_advances_a_constant_rate() {
    let rate = |_state: &[f64]| vec![2.0, -1.0];
    for result in [
        ForwardEuler.step(&[1.0, 3.0], 0.25, &rate),
        ExplicitMidpoint.step(&[1.0, 3.0], 0.25, &rate),
        RungeKutta4.step(&[1.0, 3.0], 0.25, &rate),
    ] {
        assert_eq!(result, vec![1.5, 2.75]);
    }
}

#[test]
fn every_integrator_matches_a_single_fourier_wave() {
    let n = 64;
    let k = 3;
    let c = 1.0;
    let nu = 0.05;
    let dt = 0.002;
    let steps = 100;
    let state = (0..n)
        .map(|j| exact_wave(2.0 * PI * j as f64 / n as f64, 0.0, k, c, nu))
        .collect::<Vec<_>>();
    let expected = (0..n)
        .map(|j| exact_wave(2.0 * PI * j as f64 / n as f64, dt * steps as f64, k, c, nu))
        .collect::<Vec<_>>();

    for result in [
        integrate(ForwardEuler, state.clone(), dt, steps),
        integrate(ExplicitMidpoint, state.clone(), dt, steps),
        integrate(RungeKutta4, state.clone(), dt, steps),
    ] {
        let error = result
            .iter()
            .zip(&expected)
            .map(|(actual, exact)| (actual - exact).abs())
            .fold(0.0, f64::max);
        assert!(error < 2.0e-3, "error was {error}");
    }
}

#[test]
fn fourier_rate_does_not_advect_the_nyquist_mode() {
    let n = 8;
    let rate = advection_diffusion_fourier_rate(n, 1.0, 0.05);
    let state = (0..n).map(|j| if j % 2 == 0 { 1.0 } else { -1.0 }).collect::<Vec<_>>();
    let result = rate(&state);
    let expected = -0.05 * (n as f64 / 2.0).powi(2);
    for (value, initial) in result.iter().zip(state) {
        assert!((value / initial - expected).abs() < 1.0e-10);
    }
}

#[test]
fn centered_rate_is_periodic() {
    let rate = advection_diffusion_centered_rate(4, 1.0, 0.0);
    let result = rate(&[0.0, 1.0, 0.0, -1.0]);
    assert!((result[0] + 1.0 / (2.0 * PI / 4.0)).abs() < 1.0e-12);
    assert!((result[2] - 1.0 / (2.0 * PI / 4.0)).abs() < 1.0e-12);
}
