use serde::{Deserialize, Serialize};
use std::{env, fs::{self,File}, io::{self, BufWriter, Write}, path::PathBuf, process};
use week4::{ExplicitMidpoint, ForwardEuler, Integrator, RungeKutta4, flow::SpectralFlow};

#[derive(Deserialize)]
struct Field {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[usize; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Serialize)]
struct Frame<'a> {
    t: f64,
    step: usize,
    u: &'a [f64],
    v: &'a [f64],
    omega: &'a [f64],
}

fn flag(args: &[String], name: &str) -> Result<String,String> {
    args.windows(2).find(|pair| pair[0] == name).map(|pair|pair[1].clone())
        .ok_or_else(|| format!("missing {name}"))
}
fn number<T: std::str::FromStr>(args: &[String], name: &str) -> Result<T,String> {
    flag(args,name)?.parse().map_err(|_|format!("invalid {name}"))
}
fn rounded(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| (x*1_000_000.0).round()/1_000_000.0).collect()
}

fn run() -> Result<bool,String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let method = flag(&args,"--method")?;
    if !["euler","rk2","rk4"].contains(&method.as_str()) { return Err("invalid --method".into()); }
    let nu: f64 = number(&args,"--nu")?;
    let dt: f64 = number(&args,"--dt")?;
    let t_end: f64 = number(&args,"--t-end")?;
    let every: f64 = number(&args,"--every")?;
    let out = PathBuf::from(flag(&args,"--out")?);
    if !nu.is_finite() || nu < 0.0 || !dt.is_finite() || dt <= 0.0 || !t_end.is_finite() || t_end < 0.0
        || !every.is_finite() || every <= 0.0 { return Err("invalid time or viscosity".into()); }
    let snapshot_steps = (every/dt).round() as usize;
    if snapshot_steps == 0 { return Err("--every must span at least one step".into()); }
    let input: Field = serde_json::from_reader(io::stdin()).map_err(|e|format!("field: {e}"))?;
    if input.u.len() != input.n*input.n || input.v.len() != input.n*input.n { return Err("invalid field size".into()); }
    let flow = SpectralFlow::new(input.n);
    let mut omega = flow.vorticity(&input.u,&input.v);
    fs::create_dir_all(&out).map_err(|e|e.to_string())?;
    let metadata = serde_json::json!({"case":input.case,"n":input.n,"seed":input.seed,
        "k_band":input.k_band,"method":method,"nu":nu,"dt":dt,"t_end":t_end,"snapshot_every":every});
    serde_json::to_writer_pretty(File::create(out.join("run.json")).map_err(|e|e.to_string())?,&metadata)
        .map_err(|e|e.to_string())?;
    let mut fields = BufWriter::new(File::create(out.join("fields.jsonl")).map_err(|e|e.to_string())?);
    let mut stdout = BufWriter::new(io::stdout());
    writeln!(stdout,"t\tE\tZ").map_err(|e|e.to_string())?;
    let steps = (t_end/dt).round() as usize;
    for step in 0..=steps {
        if step % snapshot_steps == 0 {
            let (u,v) = flow.velocity(&omega);
            let energy = flow.energy(&u,&v);
            let enstrophy = flow.enstrophy(&omega);
            writeln!(stdout,"{:.6}\t{:.6}\t{:.6}",step as f64*dt,energy,enstrophy).map_err(|e|e.to_string())?;
            if !energy.is_finite() || !enstrophy.is_finite() { return Ok(false); }
            let frame = Frame { t: step as f64*dt, step, u:&rounded(&u), v:&rounded(&v), omega:&rounded(&omega) };
            serde_json::to_writer(&mut fields,&frame).map_err(|e|e.to_string())?;
            writeln!(fields).map_err(|e|e.to_string())?;
        }
        if step == steps { break; }
        omega = match method.as_str() {
            "euler" => ForwardEuler.step(&omega,dt,|w|flow.rate(w,nu)),
            "rk2" => ExplicitMidpoint.step(&omega,dt,|w|flow.rate(w,nu)),
            _ => RungeKutta4.step(&omega,dt,|w|flow.rate(w,nu)),
        };
        if omega.iter().any(|x| !x.is_finite()) {
            writeln!(stdout,"{:.6}\tNaN\tNaN",(step+1) as f64*dt).map_err(|e|e.to_string())?;
            return Ok(false);
        }
    }
    Ok(true)
}

fn main() {
    match run() {
        Ok(true) => (),
        Ok(false) => process::exit(1),
        Err(error) => { eprintln!("{error}"); process::exit(2); }
    }
}
