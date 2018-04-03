use std::f64::INFINITY;

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a1, b1)| (b1 - a1).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn distance_closest_point_to_front(point: &[f64], front: &[Vec<f64>]) -> f64 {
    front
        .iter()
        .fold(INFINITY, |current_distance, front_point| {
            let distance = euclidean_distance(&point, front_point);
            if distance < current_distance {
                distance
            } else {
                current_distance
            }
        })
}

pub fn igd(approx_set: &Vec<Vec<f64>>, pareto_set: &Vec<Vec<f64>>) -> f64 {
    pareto_set
        .iter()
        .map(|pareto_point| distance_closest_point_to_front(pareto_point, &approx_set).powi(2))
        .sum::<f64>()
        .powf(1.0 / 2.0) / pareto_set.len() as f64
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
    fn finds_closest_point() {
        let point = vec![8.0, 7.0];
        let front = vec![
            vec![1.0, 5.0],
            vec![3.0, 5.0],
            vec![5.0, 4.0],
            vec![7.0, 3.0], // closest
            vec![8.0, 1.0],
        ];

        let distance = distance_closest_point_to_front(&point, &front);

        assert_eq!(distance, 17f64.sqrt());
    }

    #[test]
    fn calculates_igd_symmetric() {
        let approx_front = vec![vec![2.0, 4.0], vec![3.0, 3.0], vec![4.0, 2.0]];
        let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

        let igd_score = igd(&approx_front, &front);

        assert_approx_eq!(igd_score, (6.0_f64).sqrt() / front.len() as f64);
    }

    #[test]
    fn calculates_igd_asymmetric() {
        let approx_front = vec![
            vec![0.0, 6.0],
            vec![3.0, 4.0],
            vec![6.0, 1.0]
        ];
        let front = vec![
            vec![0.0, 4.0],
            vec![1.0, 2.0],
            vec![4.0, 0.0]
        ];

        let igd_score = igd(&approx_front, &front);

        assert_approx_eq!(igd_score, (17.0_f64).sqrt() / front.len() as f64);
    }

    #[test]
    fn calculates_igd_equal() {
        let approx_front = vec![
            vec![0.0, 6.0],
            vec![3.0, 4.0],
            vec![6.0, 1.0]
        ];
        let front = vec![
            vec![0.0, 6.0],
            vec![3.0, 4.0],
            vec![6.0, 1.0]
        ];

        let igd_score = igd(&approx_front, &front);

        assert_approx_eq!(igd_score, 0.0);
    }
}
