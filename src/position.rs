use rand::distributions::{IndependentSample, Range};
use rand::thread_rng;

pub fn random_position(lower_space: f64, upper_space: f64, dimension: usize) -> Vec<f64> {
    let between = Range::new(lower_space, upper_space);
    let mut rng = thread_rng();
    (0..dimension)
        .map(|_| between.ind_sample(&mut rng))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_random_solution() {
        let upper_space = 2.0;
        let lower_space = -4.0;
        let dimension = 4;
        let position = random_position(lower_space, upper_space, dimension);

        assert_eq!(position.len(), dimension);
        for coordinate in position {
            assert!(
                coordinate >= lower_space,
                "Coordinate({}) is outside the allowed solution space({}!",
                coordinate,
                lower_space
            );
            assert!(
                coordinate <= upper_space,
                "Coordinate({}) is outside the allowed solution space({}!",
                coordinate,
                upper_space
            );
        }
    }
}
