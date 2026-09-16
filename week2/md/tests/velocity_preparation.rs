use md::Vec2;
use md::fluid::{
    FluidError, FluidState, PeriodicBox, prepare_velocities, rescale_velocities,
    thermostat_temperature, velocity_verlet_step,
};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1.0e-12);
}

#[test]
fn preparation_removes_component_means_and_hits_target_temperature() {
    let velocities = prepare_velocities(16, 0.5, 2026).unwrap();
    let mean_x = velocities.iter().map(|velocity| velocity.x).sum::<f64>() / 16.0;
    let mean_y = velocities.iter().map(|velocity| velocity.y).sum::<f64>() / 16.0;

    assert!(mean_x.abs() < 1.0e-12);
    assert!(mean_y.abs() < 1.0e-12);
    assert_close(thermostat_temperature(&velocities), 0.5);
}

#[test]
fn same_seed_prepares_identical_velocities() {
    let first = prepare_velocities(16, 0.5, 2026).unwrap();
    let second = prepare_velocities(16, 0.5, 2026).unwrap();

    assert_eq!(first, second);
}

#[test]
fn thermostat_uses_two_n_minus_two_degrees_of_freedom() {
    let velocities = vec![
        Vec2::new(1.0, 0.0),
        Vec2::new(-1.0, 0.0),
        Vec2::new(0.0, 2.0),
        Vec2::new(0.0, -2.0),
    ];

    // E_kin = 5, so 2 E_kin / (2 N - 2) = 10 / 6.
    assert_close(thermostat_temperature(&velocities), 10.0 / 6.0);
}

#[test]
fn rescaling_reaches_target_temperature() {
    let mut velocities = vec![
        Vec2::new(1.0, 0.0),
        Vec2::new(-1.0, 0.0),
        Vec2::new(0.0, 2.0),
        Vec2::new(0.0, -2.0),
    ];

    rescale_velocities(&mut velocities, 0.5).unwrap();

    assert_close(thermostat_temperature(&velocities), 0.5);
}

#[test]
fn velocity_verlet_wraps_positions_and_preserves_free_velocity() {
    let mut state = FluidState {
        positions: vec![Vec2::new(9.9, 0.1)],
        velocities: vec![Vec2::new(1.0, -2.0)],
    };
    let box_size = PeriodicBox::new(10.0, 10.0);

    velocity_verlet_step(&mut state, &box_size, 0.2, 2.5).unwrap();

    assert_close(state.positions[0].x, 0.1);
    assert_close(state.positions[0].y, 9.7);
    assert_close(state.velocities[0].x, 1.0);
    assert_close(state.velocities[0].y, -2.0);
}

#[test]
fn velocity_verlet_updates_interacting_particles_with_opposite_velocities() {
    let mut state = FluidState {
        positions: vec![Vec2::new(2.0, 5.0), Vec2::new(4.0, 5.0)],
        velocities: vec![Vec2::ZERO, Vec2::ZERO],
    };

    velocity_verlet_step(&mut state, &PeriodicBox::new(10.0, 10.0), 0.01, 2.5).unwrap();

    assert!(state.velocities[0].x.abs() > 1.0e-6);
    assert_close(state.velocities[0].x, -state.velocities[1].x);
    assert_close(state.velocities[0].y, 0.0);
    assert_close(state.velocities[1].y, 0.0);
}

#[test]
fn velocity_verlet_leaves_state_unchanged_when_new_positions_overlap() {
    let original = FluidState {
        positions: vec![Vec2::new(2.0, 5.0), Vec2::new(5.0, 5.0)],
        velocities: vec![Vec2::new(1.0, 0.0), Vec2::new(-2.0, 0.0)],
    };
    let mut state = original.clone();

    let result = velocity_verlet_step(&mut state, &PeriodicBox::new(10.0, 10.0), 1.0, 2.5);

    assert_eq!(result, Err(FluidError::Overlap));
    assert_eq!(state, original);
}
