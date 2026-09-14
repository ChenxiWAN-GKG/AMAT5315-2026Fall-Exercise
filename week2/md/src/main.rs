use md::{EnergySample, ForwardEuler, Integrator, VelocityVerlet, initial_state, run_experiment};
use std::error::Error;

const TIME_STEP: f64 = 0.01;
const EULER_STEPS: usize = 500;
const VERLET_STEPS: usize = 5000;

fn main() -> Result<(), Box<dyn Error>> {
    let euler_samples = run_experiment(ForwardEuler, initial_state(), TIME_STEP, EULER_STEPS)?;
    let verlet_samples = run_experiment(VelocityVerlet, initial_state(), TIME_STEP, VERLET_STEPS)?;

    println!("method,step,time,total_energy,energy_error");
    for sample in &euler_samples {
        println!("{}", format_sample(ForwardEuler.name(), sample));
    }
    for sample in &verlet_samples {
        println!("{}", format_sample(VelocityVerlet.name(), sample));
    }

    Ok(())
}

fn format_sample(method: &str, sample: &EnergySample) -> String {
    format!(
        "{method},{},{:.15e},{:.15e},{:.15e}",
        sample.step, sample.time, sample.total_energy, sample.energy_error
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_one_csv_sample_with_fixed_float_precision() {
        assert_eq!(
            super::format_sample(
                "euler",
                &EnergySample {
                    step: 0,
                    time: 0.0,
                    total_energy: -1.25,
                    energy_error: 0.0,
                },
            ),
            "euler,0,0.000000000000000e0,-1.250000000000000e0,0.000000000000000e0"
        );
    }
}
