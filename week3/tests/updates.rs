use ising::Lattice;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn metropolis_step_reports_at_most_one_acceptance_per_proposal() {
    let mut lattice = Lattice::all_up(3);
    let mut rng = StdRng::seed_from_u64(2026);
    let accepted = lattice.metropolis_step(2.0, &mut rng);
    assert!(accepted <= 9);
}

#[test]
fn wolff_step_flips_a_nonempty_same_spin_cluster() {
    let mut lattice = Lattice::all_up(4);
    let before = lattice.magnetization();
    let mut rng = StdRng::seed_from_u64(2026);
    let cluster_size = lattice.wolff_step(2.0, &mut rng);
    assert!((1..=16).contains(&cluster_size));
    assert_eq!(
        lattice.magnetization(),
        before - 2.0 * cluster_size as f64 / 16.0
    );
}
