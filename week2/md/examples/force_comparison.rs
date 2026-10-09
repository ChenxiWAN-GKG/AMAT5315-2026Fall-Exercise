use md::{lennard_jones_energy, lennard_jones_force};

fn main() {
    let h = 1e-5;
    println!("r  analytic force  negative central derivative  absolute difference  tolerance");
    for r in [1.0_f64, 1.1, 1.2, 1.5] {
        let analytic = lennard_jones_force(r);
        let numerical = -(lennard_jones_energy(r + h) - lennard_jones_energy(r - h)) / (2.0 * h);
        let difference = (analytic - numerical).abs();
        let tolerance = 1e-6 * analytic.abs().max(1.0);
        println!("{r:.1}  {analytic:.12e}  {numerical:.12e}  {difference:.3e}  {tolerance:.3e}");
    }
}
