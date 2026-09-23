use std::env;

use week4::{stability_growth_factor, ExplicitMidpoint, ForwardEuler, RungeKutta4};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let value = |name: &str| {
        args.windows(2)
            .find(|pair| pair[0] == name)
            .expect("missing argument")[1]
            .parse::<f64>()
            .expect("invalid number")
    };
    let xmin = value("--xmin");
    let xmax = value("--xmax");
    let ymin = value("--ymin");
    let ymax = value("--ymax");
    let points = value("--points") as usize;
    println!("i,j,re,im,euler,midpoint,rk4");
    for j in 0..points {
        let im = ymin + (ymax - ymin) * j as f64 / (points - 1) as f64;
        for i in 0..points {
            let re = xmin + (xmax - xmin) * i as f64 / (points - 1) as f64;
            println!("{i},{j},{re:.15e},{im:.15e},{:.15e},{:.15e},{:.15e}",
                stability_growth_factor(ForwardEuler, re, im),
                stability_growth_factor(ExplicitMidpoint, re, im),
                stability_growth_factor(RungeKutta4, re, im));
        }
    }
}
