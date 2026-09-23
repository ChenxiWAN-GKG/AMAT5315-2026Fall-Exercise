use std::env;
use std::f64::consts::PI;

use week4::{Integrator, RungeKutta4, advection_diffusion_fourier_rate};

fn argument(name: &str) -> f64 {
    env::args().skip(1).collect::<Vec<_>>().windows(2)
        .find(|pair| pair[0] == name)
        .expect("missing argument")[1]
        .parse().expect("invalid argument")
}

fn main() {
    let n = argument("--n") as usize;
    let c = argument("--c");
    let nu = argument("--nu");
    let dt = argument("--dt");
    let final_time = argument("--final-time");
    let sigma = argument("--sigma");
    let rate = advection_diffusion_fourier_rate(n, c, nu);
    let mut state = (0..n).map(|j| {
        let x = 2.0 * PI * j as f64 / n as f64;
        (-3..=3).map(|image| {
            let distance = x - PI / 2.0 + 2.0 * PI * image as f64;
            (-distance * distance / (2.0 * sigma * sigma)).exp()
        }).sum()
    }).collect::<Vec<_>>();
    let steps = (final_time / dt).round() as usize;
    println!("t,j,x,u");
    for (j, value) in state.iter().enumerate() {
        let x = 2.0 * PI * j as f64 / n as f64;
        println!("0.0,{j},{x},{value}");
    }
    for step in 1..=steps {
        state = RungeKutta4.step(&state, dt, &rate);
        for (j, value) in state.iter().enumerate() {
            let x = 2.0 * PI * j as f64 / n as f64;
            println!("{},{},{},{}", step as f64 * dt, j, x, value);
        }
    }
}
