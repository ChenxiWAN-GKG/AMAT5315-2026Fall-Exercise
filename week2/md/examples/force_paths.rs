use md::Vec2;
use md::fluid::{FluidState, ForceMethod, PeriodicBox, forces_and_energy_with_method, triangular_lattice};

fn report(label: &str, box_size: PeriodicBox, positions: Vec<Vec2>) {
    let state = FluidState {
        velocities: vec![Vec2::ZERO; positions.len()],
        positions,
    };
    let (naive, e_naive) = forces_and_energy_with_method(&state, &box_size, 2.5, ForceMethod::Naive).unwrap();
    let (cells, e_cells) = forces_and_energy_with_method(&state, &box_size, 2.5, ForceMethod::Cells).unwrap();
    let max_force_difference = naive.iter().zip(&cells).map(|(a, b)| (a.x - b.x).abs().max((a.y - b.y).abs())).fold(0.0_f64, f64::max);
    println!("{label}: N={}, tolerance=1e-10", naive.len());
    println!("  naive energy={e_naive:.12e}; cells energy={e_cells:.12e}; |difference|={:.3e}", (e_naive - e_cells).abs());
    println!("  naive forces={:?}", naive);
    println!("  cells forces={:?}", cells);
    println!("  max component force difference={max_force_difference:.3e}");
}

fn main() {
    let (box_size, mut lattice) = triangular_lattice(100, 0.8).unwrap();
    for (i, point) in lattice.iter_mut().enumerate() {
        point.x = (point.x + 0.01 * (i as f64).sin()).rem_euclid(box_size.lx);
        point.y = (point.y + 0.01 * (i as f64).cos()).rem_euclid(box_size.ly);
    }
    report("perturbed 100-atom lattice", box_size, lattice);
    report("boundary-crossing pair", PeriodicBox::new(10.0, 10.0), vec![Vec2::new(0.2, 0.2), Vec2::new(9.5, 0.2)]);
    report("cutoff pair", PeriodicBox::new(10.0, 10.0), vec![Vec2::new(0.0, 0.0), Vec2::new(2.5, 0.0)]);
    report("two-cell-wide box", PeriodicBox::new(5.1, 5.1), vec![Vec2::new(0.1, 0.1), Vec2::new(2.4, 0.1), Vec2::new(4.9, 5.0), Vec2::new(2.5, 2.5)]);
}
