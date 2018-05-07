use multiobjective::domination::find_non_dominated_n_objectives;
use operators::position::euclidean_distance;
use statistics::fronts::invert_normalized_front;
use std::f64::INFINITY;

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

pub fn gd(approx_set: &Vec<Vec<f64>>, pareto_set: &Vec<Vec<f64>>) -> f64 {
    approx_set
        .iter()
        .map(|pareto_point| distance_closest_point_to_front(pareto_point, &pareto_set).powi(2))
        .sum::<f64>()
        .powf(1.0 / 2.0) / approx_set.len() as f64
}

fn surface_unchanged_to(front: &Vec<Vec<f64>>, objective: usize) -> f64 {
    let mut min = front[0][objective];
    for i in 1..front.len() {
        let value = front[i][objective];
        if value < min {
            min = value;
        }
    }
    min
}

fn reduce_non_dominated_set(
    front: Vec<Vec<f64>>,
    objective: usize,
    treshhold: f64,
) -> Vec<Vec<f64>> {
    front
        .into_iter()
        .filter(|point| point[objective] > treshhold)
        .collect()
}

pub fn hyper_volume(front: &Vec<Vec<f64>>) -> f64 {
    let objectives = front[0].len();
    let front = invert_normalized_front(&front);
    hyper_volume_dimension(&front, objectives)
}

fn hyper_volume_dimension(front: &Vec<Vec<f64>>, objectives: usize) -> f64 {
    let mut volume = 0.0;
    let mut distance = 0.0;
    let mut front = front.clone();
    while front.len() > 0 {
        let non_dominated_indices = find_non_dominated_n_objectives(&front, objectives - 1, true);
        let temp_volume = if objectives < 3 {
            front[0][0]
        } else {
            let front = front
                .clone()
                .into_iter()
                .enumerate()
                .filter(|(i, _)| non_dominated_indices.contains(i))
                .map(|(_, s)| s)
                .collect();
            hyper_volume_dimension(&front, objectives - 1)
        };
        let temp_distance = surface_unchanged_to(&front, objectives - 1);
        volume += temp_volume * (temp_distance - distance);
        distance = temp_distance;
        front = reduce_non_dominated_set(front, objectives - 1, distance);
    }
    volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;
    use statistics::fronts::{front_min_max, normalize_front};
    use std::fs::File;

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
        let approx_front = vec![vec![0.0, 6.0], vec![3.0, 4.0], vec![6.0, 1.0]];
        let front = vec![vec![0.0, 4.0], vec![1.0, 2.0], vec![4.0, 0.0]];

        let igd_score = igd(&approx_front, &front);

        assert_approx_eq!(igd_score, (17.0_f64).sqrt() / front.len() as f64);
    }

    #[test]
    fn calculates_igd_equal() {
        let approx_front = vec![vec![0.0, 6.0], vec![3.0, 4.0], vec![6.0, 1.0]];
        let front = vec![vec![0.0, 6.0], vec![3.0, 4.0], vec![6.0, 1.0]];

        let igd_score = igd(&approx_front, &front);

        assert_approx_eq!(igd_score, 0.0);
    }

    #[test]
    fn calculates_hyper_volume_zdt1() {
        let file = File::open("optimal_solutions/zdt1-2d.json").unwrap();
        let front: Vec<Vec<f64>> = serde_json::from_reader(file).unwrap();
        let minmax = front_min_max(&front);
        let front = normalize_front(&front, &minmax);

        let volume = hyper_volume(&front);

        assert_eq!(volume, 0.6661601248750002);
    }

    #[test]
    fn calculates_hyper_volume_dtlz1_2d() {
        let file = File::open("optimal_solutions/dtlz1-2d.json").unwrap();
        let front: Vec<Vec<f64>> = serde_json::from_reader(file).unwrap();
        let minmax = front_min_max(&front);

        let volume = hyper_volume(&front);

        assert_eq!(volume, 0.8748748748248245);
    }

    #[test]
    fn calculates_gd_symmetric() {
        let approx_front = vec![vec![2.0, 4.0], vec![3.0, 3.0], vec![4.0, 2.0]];
        let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];

        let gd_score = gd(&approx_front, &front);

        assert_approx_eq!(gd_score, (6.0_f64).sqrt() / front.len() as f64);
    }

    #[test]
    fn calculates_gd_asymmetric() {
        let approx_front = vec![vec![0.0, 6.0], vec![3.0, 4.0], vec![6.0, 1.0]];
        let front = vec![vec![0.0, 4.0], vec![1.0, 2.0], vec![4.0, 0.0]];

        let gd_score = gd(&approx_front, &front);

        assert_approx_eq!(gd_score, (17.0_f64).sqrt() / front.len() as f64);
    }

    #[test]
    fn calculates_gd_equal() {
        let approx_front = vec![vec![0.0, 6.0], vec![3.0, 4.0], vec![6.0, 1.0]];
        let front = vec![vec![0.0, 6.0], vec![3.0, 4.0], vec![6.0, 1.0]];

        let gd_score = gd(&approx_front, &front);

        assert_approx_eq!(gd_score, 0.0);
    }
}
