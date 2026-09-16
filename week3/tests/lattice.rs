use ising::Lattice;

#[test]
fn all_up_two_by_two_has_magnetization_one_and_energy_minus_two() {
    let lattice = Lattice::all_up(2);
    assert_eq!(lattice.magnetization(), 1.0);
    assert_eq!(lattice.energy_per_site(), -2.0);
}

#[test]
fn local_flip_energy_matches_the_total_energy_change() {
    let mut lattice = Lattice::all_up(3);
    let before = lattice.energy_per_site();
    let delta = lattice.delta_energy(0) as f64;
    lattice.flip(0);
    let after = lattice.energy_per_site();
    assert!((after - before) * 9.0 == delta);
}
