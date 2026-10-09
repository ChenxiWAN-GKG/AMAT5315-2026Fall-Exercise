use std::f64::consts::PI;
use week4::{Integrator, RungeKutta4, flow::SpectralFlow};

#[test]
fn represented_wave_has_exact_derivatives() {
    let flow = SpectralFlow::new(32);
    let g = (0..32).flat_map(|j| (0..32).map(move |i| {
        (3.0 * 2.0 * PI * i as f64 / 32.0).sin()
            * (2.0 * 2.0 * PI * j as f64 / 32.0).cos()
    })).collect::<Vec<_>>();
    let dx = flow.derivative(&g, 1, 0);
    let dxx = flow.derivative(&g, 2, 0);
    let dxdy = flow.derivative(&g, 1, 1);
    let lap = flow.laplacian(&g);
    for j in 0..32 {
        for i in 0..32 {
            let x = 2.0 * PI * i as f64 / 32.0;
            let y = 2.0 * PI * j as f64 / 32.0;
            let p = j * 32 + i;
            assert!((dx[p] - 3.0 * (3.0*x).cos() * (2.0*y).cos()).abs() < 1e-10);
            assert!((dxx[p] + 9.0*g[p]).abs() < 1e-10);
            assert!((dxdy[p] + 6.0*(3.0*x).cos()*(2.0*y).sin()).abs() < 1e-10);
            assert!((lap[p] + 13.0*g[p]).abs() < 1e-10);
        }
    }
}

#[test]
fn taylor_green_decays_at_its_exact_rate() {
    let n = 16;
    let flow = SpectralFlow::new(n);
    let mut omega = (0..n).flat_map(|j| (0..n).map(move |i| {
        -2.0 * (2.0*PI*i as f64/n as f64).cos()
            * (2.0*PI*j as f64/n as f64).cos()
    })).collect::<Vec<_>>();
    for _ in 0..100 {
        omega = RungeKutta4.step(&omega, 0.01, |w| flow.rate(w, 0.1));
    }
    let expected = (-0.2_f64).exp();
    let exact = -2.0 * expected;
    assert!((omega[0] - exact).abs() < 1e-8);
    let (u, v) = flow.velocity(&omega);
    let energy = flow.energy(&u, &v);
    assert!((energy - 0.25*(-0.4_f64).exp()).abs() < 1e-8);
}
