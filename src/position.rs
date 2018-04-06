use rand::distributions::{IndependentSample, Range};
use rand::thread_rng;

pub fn random_position(lower_space: f64, upper_space: f64, dimension: usize) -> Vec<f64> {
    let between = Range::new(lower_space, upper_space);
    let mut rng = thread_rng();
    (0..dimension)
        .map(|_| between.ind_sample(&mut rng))
        .collect()
}

// https://mathoverflow.net/a/87691
// Creates perpendicular vector with euclidean norm of 1
pub fn perpendicular_position(position: &Vec<f64>) -> Vec<f64> {
    let mut perpendicular = vec![0.0; position.len()];
    if position[0] == 0.0 {
        perpendicular[0] = 1.0;
    } else if position[1] == 0.0 {
        perpendicular[1] = 1.0;
    } else {
        let x1 = position[0];
        let x2 = position[1];
        let norm = (x1.powi(2) + x2.powi(2)).sqrt();
        perpendicular[0] = -x2 / norm;
        perpendicular[1] = x1 / norm;
    }
    perpendicular
}

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a1, b1)| (b1 - a1).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_distance_correct() {
        let a = vec![1.0, 0.0];
        let b = vec![4.0, 3.0];

        let distance = euclidean_distance(&a, &b);

        assert_eq!(distance, 18f64.sqrt());
    }

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

    #[test]
    fn finds_perpendicular_vector_ones() {
        let position = vec![1.0, 1.0];

        let perpendicular = perpendicular_position(&position);

        assert_eq!(
            perpendicular,
            vec![-1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()]
        );
    }

    #[test]
    fn finds_perpendicular_vector_higher_values() {
        let position = vec![2.0, 3.0];

        let perpendicular = perpendicular_position(&position);

        assert_eq!(
            perpendicular,
            vec![-3.0 / 13.0f64.sqrt(), 2.0 / 13.0f64.sqrt()]
        );
    }

    #[test]
    fn finds_perpendicular_vector_first_zero() {
        let position = vec![0.0, 1.0, 0.0];

        let perpendicular = perpendicular_position(&position);

        assert_eq!(perpendicular, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn finds_perpendicular_vector_second_zero() {
        let position = vec![1.0, 0.0, 0.0];

        let perpendicular = perpendicular_position(&position);

        assert_eq!(perpendicular, vec![0.0, 1.0, 0.0]);
    }
}
