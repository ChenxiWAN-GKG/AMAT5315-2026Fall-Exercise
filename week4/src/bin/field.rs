use serde::Serialize;
use std::{env, f64::consts::PI, process};
use week4::flow::SpectralFlow;

#[derive(Serialize)]
struct Field {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[usize; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].clone())
}

fn required<T: std::str::FromStr>(args: &[String], name: &str) -> T {
    flag(args, name).and_then(|value| value.parse().ok()).unwrap_or_else(|| {
        eprintln!("missing or invalid {name}"); process::exit(2)
    })
}

fn random_phase(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    2.0*PI*((z >> 11) as f64)/((1_u64 << 53) as f64)
}

fn run() -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let case = args.first().ok_or("use field taylor-green|random --n N ...")?.as_str();
    let n: usize = required(&args, "--n");
    if n < 8 || n % 2 != 0 { return Err("--n must be even and at least 8".into()); }
    let flow = SpectralFlow::new(n);
    let field = match case {
        "taylor-green" => {
            let t: f64 = flag(&args,"--t").unwrap_or_else(|| "0".into()).parse().map_err(|_| "invalid --t")?;
            let nu: f64 = if t == 0.0 { flag(&args,"--nu").unwrap_or_else(|| "0".into()).parse().map_err(|_| "invalid --nu")? }
                else { required(&args,"--nu") };
            let (u,v) = flow.taylor_green(nu,t);
            Field { case: case.into(), n, seed: None, k_band: None, u, v }
        }
        "random" => {
            let seed: u64 = required(&args,"--seed");
            let k_min: usize = required(&args,"--k-min");
            let k_max: usize = required(&args,"--k-max");
            if k_min == 0 || k_min > k_max || k_max > n/3 { return Err("invalid wavenumber band".into()); }
            let mut state = seed;
            let mut modes = Vec::new();
            for ky in -(k_max as i32)..=(k_max as i32) {
                for kx in -(k_max as i32)..=(k_max as i32) {
                    let k2 = (kx*kx+ky*ky) as usize;
                    if (kx > 0 || (kx == 0 && ky > 0)) && k2 >= k_min*k_min && k2 <= k_max*k_max {
                        modes.push((kx as f64,ky as f64,random_phase(&mut state)));
                    }
                }
            }
            let omega = (0..n).flat_map(|j| {
                let modes = &modes;
                (0..n).map(move |i| {
                    let x = 2.0*PI*i as f64/n as f64;
                    let y = 2.0*PI*j as f64/n as f64;
                    modes.iter().map(|(kx,ky,phase)| (kx*x+ky*y+phase).cos()).sum::<f64>()
                })
            }).collect::<Vec<_>>();
            let (mut u,mut v) = flow.velocity(&omega);
            let scale = (0.5/flow.energy(&u,&v)).sqrt();
            for value in &mut u { *value *= scale; }
            for value in &mut v { *value *= scale; }
            Field { case: case.into(), n, seed: Some(seed), k_band: Some([k_min,k_max]), u, v }
        }
        _ => return Err("use field taylor-green|random --n N ...".into()),
    };
    println!("{}",serde_json::to_string(&field).map_err(|e| e.to_string())?);
    Ok(())
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); process::exit(2); }
}
