use md::fluid::{FluidState, PeriodicBox, shifted_lj_potential, total_forces};
use md::Vec2;

fn test_state() -> FluidState {
    FluidState {
        positions: vec![
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(1.5, 1.0 + 3.0_f64.sqrt() / 2.0),
        ],
        velocities: vec![Vec2::ZERO; 3],
    }
}

#[test]
fn total_force_over_all_atoms_is_zero() {
    let system = test_state();
    let forces = total_forces(&system, &PeriodicBox::new(8.0, 8.0), 2.5).unwrap();
    let sum = forces.iter().fold(Vec2::ZERO, |acc, force| {
        Vec2::new(acc.x + force.x, acc.y + force.y)
    });
    assert!(sum.x.abs() < 1.0e-12);
    assert!(sum.y.abs() < 1.0e-12);
}

#[test]
fn shifted_potential_is_continuous_at_cutoff_from_inside() {
    let rc = 2.5;
    let epsilon = 1.0e-7;
    let inside = shifted_lj_potential(rc - epsilon, rc);
    let at_cutoff = shifted_lj_potential(rc, rc);
    let outside = shifted_lj_potential(rc + epsilon, rc);
    assert!(inside.abs() < 1.0e-5);
    assert_eq!(at_cutoff, 0.0);
    assert_eq!(outside, 0.0);
}
