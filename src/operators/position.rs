use rand::distributions::{IndependentSample, Range};
use rand::{weak_rng, Rng};

pub fn random_position(lower_space: f64, upper_space: f64, dimensions: usize) -> Vec<f64> {
    let between = Range::new(lower_space, upper_space);
    let mut rng = weak_rng();
    (0..dimensions)
        .map(|_| between.ind_sample(&mut rng))
        .collect()
}

pub fn multi_random_position(lower_bound: &Vec<f64>, upper_bound: &Vec<f64>) -> Vec<f64> {
    let mut rng = weak_rng();
    (0..upper_bound.len())
        .map(|i| rng.gen_range::<f64>(lower_bound[i], upper_bound[i]))
        .collect()
}

fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2).map(|(v1, v2)| v1 * v2).sum()
}

fn approx_zero(value: f64) -> bool {
    let eps = 0.000001;
    (value >= 0.0 && value < eps) || (value < 0.0 && value > -eps)
}

// Creates perpendicular vector with euclidean norm of 1
pub fn perpendicular_position(position: &Vec<f64>, mut rng: impl Rng) -> Vec<f64> {
    // Create random vector with values -1.0 to 1.0
    let mut perpendicular = position.iter().map(|_| rng.gen_range(-1.0, 1.0)).collect();
    // Loop until dot product is 0 (vector is perpendicular)
    while !approx_zero(dot_product(&position, &perpendicular)) {
        let dot = dot_product(&position, &perpendicular);
        // Random dimension
        let d = rng.gen_range(0, position.len());

        let pos_d = position[d];
        let perp_d = perpendicular[d];
        let dot_d = pos_d * perp_d;

        // Calculate new value
        let mut new_perp_d = -(dot - dot_d) / (pos_d);

        // Limit to either -1.0 or 1.0
        if new_perp_d < -1.0 {
            new_perp_d = -1.0;
        } else if new_perp_d > 1.0 {
            new_perp_d = 1.0;
        }

        perpendicular[d] = new_perp_d;
    }
    // Normalize vector (||v|| = 1)
    let norm = (perpendicular.iter().map(|p| p.powi(2)).sum::<f64>()).sqrt();
    perpendicular.iter().map(|p| p / norm).collect()
}

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a1, b1)| (b1 - a1).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn limit_position(position: &mut Vec<f64>, lower_bound: f64, upper_bound: f64) {
    for i in 0..position.len() {
        let value = position[i];
        if value > upper_bound {
            position[i] = upper_bound.clone();
        } else if value < lower_bound {
            position[i] = lower_bound.clone();
        }
    }
}

pub fn limit_position_random(position: &mut Vec<f64>, lower_bound: f64, upper_bound: f64) {
    let out_of_bounds = position
        .iter()
        .any(|&x_i| x_i > upper_bound || x_i < lower_bound);
    if out_of_bounds {
        *position = random_position(lower_bound, upper_bound, position.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use testing::utils::create_rng;

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
        let dimensions = 4;
        let position = random_position(lower_space, upper_space, dimensions);

        assert_eq!(position.len(), dimensions);
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
    fn finds_perpendicular_vector_simple() {
        let position = vec![1.0, 0.0];

        let perpendicular = perpendicular_position(&position, create_rng());

        assert_approx_eq!(dot_product(&position, &perpendicular), 0.0);
        assert_approx_eq!(perpendicular.iter().map(|a| a.powi(2)).sum::<f64>(), 1.0);
    }

    #[test]
    fn finds_perpendicular_vector() {
        let position = vec![2.0, 3.0, 4.0, 2.0, 3.0];

        let perpendicular = perpendicular_position(&position, create_rng());

        assert_approx_eq!(dot_product(&position, &perpendicular), 0.0);
        assert_approx_eq!(perpendicular.iter().map(|a| a.powi(2)).sum::<f64>(), 1.0);
    }

    #[test]
    fn finds_perpendicular_vector_equal() {
        let position = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

        let perpendicular = perpendicular_position(&position, create_rng());

        assert_approx_eq!(dot_product(&position, &perpendicular), 0.0);
        assert_approx_eq!(perpendicular.iter().map(|a| a.powi(2)).sum::<f64>(), 1.0);
    }

    #[test]
    fn test_limit_position() {
        let mut position = vec![0.5, 1.0, 10.0, -5.0, -3.0, 4.0, -10.0];

        limit_position(&mut position, -4.0, 4.0);

        assert_eq!(position, vec![0.5, 1.0, 4.0, -4.0, -3.0, 4.0, -4.0]);
    }

    #[test]
    fn test_limit_position_random_out_of_bounds() {
        let mut position = vec![0.5, 1.0, 10.0, -5.0, -3.0, 4.0, -10.0];
        let original = position.clone();

        limit_position_random(&mut position, -4.0, 4.0);

        assert_ne!(position, original);
    }

    #[test]
    fn test_limit_position_random() {
        let mut position = vec![0.5, 1.0, 2.0, -3.9, -3.0, 4.0, -1.0];
        let original = position.clone();

        limit_position_random(&mut position, -4.0, 4.0);

        assert_eq!(position, original);
    }

    #[bench]
    #[ignore]
    fn bench_perpendicular(b: &mut Bencher) {
        let position = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut rng = weak_rng();
        b.iter(|| perpendicular_position(&position, &mut rng));
    }
}
