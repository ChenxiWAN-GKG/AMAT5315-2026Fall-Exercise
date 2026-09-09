/// Return the program's greeting.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

#[cfg(test)]
mod tests {
    use super::greeting;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn greeting_says_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn lennard_jones_pair_energy_at_distance_two() {
        let energy = super::lennard_jones_energy(2.0);
        let expected_energy = -63.0 / 1024.0;

        assert!((energy - expected_energy).abs() < TOLERANCE);
    }

    #[test]
    fn lennard_jones_pair_force_at_distance_two() {
        let force = super::lennard_jones_force(2.0);
        let expected_force = -93.0 / 512.0;

        assert!((force - expected_force).abs() < TOLERANCE);
    }
}
