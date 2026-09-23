use std::env;
use std::f64::consts::PI;

use week4::{
    ForwardEuler, Integrator, RungeKutta4, advection_diffusion_centered_rate,
    advection_diffusion_fourier_rate,
};

fn argument(name: &str) -> String {
    env::args().skip(1).collect::<Vec<_>>().windows(2)
        .find(|pair| pair[0] == name)
        .expect("missing argument")[1]
        .clone()
}

fn number(name: &str) -> f64 {
    argument(name).parse().expect("invalid numeric argument")
}

fn periodic_gaussian(n: usize, sigma: f64, center: f64) -> Vec<f64> {
    (0..n).map(|j| {
        let x = 2.0 * PI * j as f64 / n as f64;
        (-3..=3).map(|image| {
            let distance = x - center + 2.0 * PI * image as f64;
            (-distance * distance / (2.0 * sigma * sigma)).exp()
        }).sum()
    }).collect()
}

fn main() {
    let n = number("--n") as usize;
    let c = number("--c");
    let nu = number("--nu");
    let dt = number("--dt");
    let final_time = number("--final-time");
    let sigma = number("--sigma");
    let derivative = argument("--derivative");
    let integrator = argument("--integrator");
    let rate = if derivative == "fourier" {
        Box::new(advection_diffusion_fourier_rate(n, c, nu)) as Box<dyn Fn(&[f64]) -> Vec<f64>>
    } else if derivative == "centered" {
        Box::new(advection_diffusion_centered_rate(n, c, nu))
    } else {
        panic!("unknown derivative: {derivative}");
    };
    let mut state = periodic_gaussian(n, sigma, PI / 2.0);
    let mut time = 0.0;
    while time < final_time {
        let step = dt.min(final_time - time);
        state = if integrator == "euler" {
            ForwardEuler.step(&state, step, &rate)
        } else if integrator == "rk4" {
            RungeKutta4.step(&state, step, &rate)
        } else {
            panic!("unknown integrator: {integrator}");
        };
        time += step;
    }
    println!("j,x,u");
    for (j, value) in state.iter().enumerate() {
        let x = 2.0 * PI * j as f64 / n as f64;
        println!("{j},{x},{value}");
    }
}
