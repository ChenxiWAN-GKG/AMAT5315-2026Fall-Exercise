use md::Vec2;
use md::fluid::{FluidState, ForceMethod, PeriodicBox, forces_and_energy_with_method};

#[test]
fn cell_list_matches_naive_for_reachable_pair_geometries() {
    let cases = [
        (PeriodicBox::new(10.0, 10.0), vec![[0.2, 0.3], [1.3, 0.3], [9.5, 0.4], [4.0, 4.0]]),
        (PeriodicBox::new(5.1, 5.1), vec![[0.1, 0.1], [2.4, 0.1], [4.9, 5.0], [2.5, 2.5]]),
        (PeriodicBox::new(10.0, 10.0), vec![[0.0, 0.0], [2.5, 0.0], [2.499, 1.0], [7.0, 7.0]]),
    ];
    for (box_size, points) in cases {
        let state = FluidState {
            positions: points.into_iter().map(|p| Vec2::new(p[0], p[1])).collect(),
            velocities: vec![Vec2::ZERO; 4],
        };
        let (naive, naive_energy) = forces_and_energy_with_method(&state, &box_size, 2.5, ForceMethod::Naive).unwrap();
        let (cells, cells_energy) = forces_and_energy_with_method(&state, &box_size, 2.5, ForceMethod::Cells).unwrap();
        assert!((naive_energy - cells_energy).abs() < 1e-10);
        for (a, b) in naive.iter().zip(&cells) {
            assert!((a.x - b.x).abs() < 1e-10);
            assert!((a.y - b.y).abs() < 1e-10);
        }
    }
}
