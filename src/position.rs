use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng};

pub fn random_position(lower_space: f64, upper_space: f64, dimension: usize) -> Vec<f64> {
    let between = Range::new(lower_space, upper_space);
    let mut rng = thread_rng();
    (0..dimension)
        .map(|_| between.ind_sample(&mut rng))
        .collect()
}

fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2).map(|(v1, v2)| v1 * v2).sum()
}

fn approx_equal(a: f64, b: f64) -> bool {
    let c = b - a;
    let eps = 0.001;
    (c < eps && c >= 0.0) || (c > -eps && c <= 0.0)
}

// Creates perpendicular vector with euclidean norm of 1
pub fn perpendicular_position(position: &Vec<f64>, mut rng: impl Rng) -> Vec<f64> {
    let mut perpendicular = position.iter().map(|_| rng.gen_range(-1.0, 1.0)).collect();
    while !approx_equal(dot_product(&position, &perpendicular), 0.0) {
        let dot = dot_product(&position, &perpendicular);
        let d = rng.gen_range(0, position.len());
        let dot_d = (position[d] * perpendicular[d]);
        let perp_d = perpendicular[d];
        if (dot > 0.0) == ((dot_d > 0.0) == (perp_d > 0.0)) {
            perpendicular[d] = rng.gen_range(-1.0, perpendicular[d]);
        } else {
            perpendicular[d] = rng.gen_range(perpendicular[d], 1.0);
        }
    }
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
    fn finds_perpendicular_vector() {
        let position = vec![2.0, 3.0, 4.0, 2.0, 3.0];

        let perpendicular = perpendicular_position(&position, create_rng());

        assert_eq!(
            perpendicular,
            vec![
                -0.4510884382825031,
                -0.48170017037179613,
                0.5052849241820805,
                0.506681167501091,
                -0.22900983953899332,
            ]
        );
    }

    #[test]
    fn finds_perpendicular_vector_equal() {
        let position = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

        let perpendicular = perpendicular_position(&position, create_rng());

        assert_eq!(
            perpendicular,
            vec![
                0.16556261327398303,
                -0.47293524388015784,
                -0.11411503808514509,
                0.4549193955638799,
                0.49722485154735896,
                -0.5307681459999574,
            ]
        );
    }

    #[bench]
    #[ignore]
    fn bench_perpendicular(b: &mut Bencher) {
        let position = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut rng = thread_rng();
        b.iter(|| perpendicular_position(&position, &mut rng));
    }
}
