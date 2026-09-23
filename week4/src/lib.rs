//! Time integrators and periodic advection-diffusion rates for Week 4.

use std::f64::consts::PI;

/// One explicit time step for a vector-valued differential equation.
pub trait Integrator {
    /// Advance `state` by `dt` using the supplied rate function.
    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardEuler;

impl Integrator for ForwardEuler {
    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let derivative = rate(state);
        state.iter().zip(derivative).map(|(y, dy)| y + dt * dy).collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitMidpoint;

impl Integrator for ExplicitMidpoint {
    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(state);
        let midpoint = state.iter().zip(&k1).map(|(y, dy)| y + 0.5 * dt * dy).collect::<Vec<_>>();
        let k2 = rate(&midpoint);
        state.iter().zip(k2).map(|(y, dy)| y + dt * dy).collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RungeKutta4;

impl Integrator for RungeKutta4 {
    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(state);
        let stage2 = state.iter().zip(&k1).map(|(y, dy)| y + 0.5 * dt * dy).collect::<Vec<_>>();
        let k2 = rate(&stage2);
        let stage3 = state.iter().zip(&k2).map(|(y, dy)| y + 0.5 * dt * dy).collect::<Vec<_>>();
        let k3 = rate(&stage3);
        let stage4 = state.iter().zip(&k3).map(|(y, dy)| y + dt * dy).collect::<Vec<_>>();
        let k4 = rate(&stage4);
        state.iter().zip(k1).zip(k2).zip(k3).zip(k4)
            .map(|((((y, a), b), c), d)| y + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EqualWeightRungeKutta4;

impl Integrator for EqualWeightRungeKutta4 {
    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(state);
        let stage2 = state.iter().zip(&k1).map(|(y, dy)| y + 0.5 * dt * dy).collect::<Vec<_>>();
        let k2 = rate(&stage2);
        let stage3 = state.iter().zip(&k2).map(|(y, dy)| y + 0.5 * dt * dy).collect::<Vec<_>>();
        let k3 = rate(&stage3);
        let stage4 = state.iter().zip(&k3).map(|(y, dy)| y + dt * dy).collect::<Vec<_>>();
        let k4 = rate(&stage4);
        state.iter().zip(k1).zip(k2).zip(k3).zip(k4)
            .map(|((((y, a), b), c), d)| y + dt * (a + b + c + d) / 4.0)
            .collect()
    }
}

/// Return the Fourier spectral rate for `u_t + c u_x = nu u_xx`.
pub fn advection_diffusion_fourier_rate(
    n: usize,
    c: f64,
    nu: f64,
) -> impl Fn(&[f64]) -> Vec<f64> {
    move |state| {
        assert_eq!(state.len(), n, "state length must equal n");
        let mut result = vec![0.0; n];
        for k_index in 0..n {
            let k = if k_index <= n / 2 - 1 {
                k_index as isize
            } else {
                k_index as isize - n as isize
            };
            let derivative_multiplier = if k_index == n / 2 {
                0.0
            } else {
                k as f64
            };
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (j, value) in state.iter().enumerate() {
                let angle = -2.0 * PI * k_index as f64 * j as f64 / n as f64;
                real += value * angle.cos();
                imaginary += value * angle.sin();
            }
            real /= n as f64;
            imaginary /= n as f64;
            let diffusion_multiplier = -(k as f64).powi(2);
            let rate_real = nu * diffusion_multiplier * real + c * derivative_multiplier * imaginary;
            let rate_imaginary = nu * diffusion_multiplier * imaginary - c * derivative_multiplier * real;
            for (j, output) in result.iter_mut().enumerate() {
                let angle = 2.0 * PI * k_index as f64 * j as f64 / n as f64;
                *output += rate_real * angle.cos() - rate_imaginary * angle.sin();
            }
        }
        result
    }
}

/// Return the periodic centred-finite-difference rate for the same equation.
pub fn advection_diffusion_centered_rate(
    n: usize,
    c: f64,
    nu: f64,
) -> impl Fn(&[f64]) -> Vec<f64> {
    move |state| {
        assert_eq!(state.len(), n, "state length must equal n");
        let dx = 2.0 * PI / n as f64;
        (0..n).map(|j| {
            let left = state[(j + n - 1) % n];
            let center = state[j];
            let right = state[(j + 1) % n];
            -c * (right - left) / (2.0 * dx) + nu * (right - 2.0 * center + left) / dx.powi(2)
        }).collect()
    }
}

/// Advance the real two-component representation of `y' = lambda y` once.
pub fn rk4_complex_growth_factor(real_lambda: f64, imaginary_lambda: f64) -> f64 {
    let rate = |state: &[f64]| {
        vec![
            real_lambda * state[0] - imaginary_lambda * state[1],
            imaginary_lambda * state[0] + real_lambda * state[1],
        ]
    };
    let result = RungeKutta4.step(&[1.0, 0.0], 1.0, rate);
    result.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub fn stability_growth_factor(integrator: impl Integrator, real_lambda: f64, imaginary_lambda: f64) -> f64 {
    let rate = |state: &[f64]| {
        vec![
            real_lambda * state[0] - imaginary_lambda * state[1],
            imaginary_lambda * state[0] + real_lambda * state[1],
        ]
    };
    let result = integrator.step(&[1.0, 0.0], 1.0, rate);
    result.iter().map(|value| value * value).sum::<f64>().sqrt()
}
