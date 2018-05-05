use itertools::Itertools;
use itertools::MinMaxResult;

pub fn front_min_max(front: &Vec<Vec<f64>>) -> Vec<(f64, f64)> {
    (0..front[0].len())
        .map(|objective| {
            let minmax = front.iter().map(|point| point[objective]).minmax();
            if let MinMaxResult::MinMax(min, max) = minmax {
                return (min, max);
            } else {
                panic!("Unable to find min max for front");
            }
        })
        .collect()
}

pub fn normalize_front(front: &Vec<Vec<f64>>, minmax: &Vec<(f64, f64)>) -> Vec<Vec<f64>> {
    front
        .iter()
        .map(|point| {
            point
                .iter()
                .enumerate()
                .map(|(objective, x)| {
                    let (min, max) = minmax[objective];
                    (x - min) / (max - min)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_min_max() {
        let front = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.2, 4.3],
            vec![0.6, 0.1, 1.3],
        ];

        let min_max = front_min_max(&front);

        assert_eq!(min_max, vec![(0.1, 0.6), (0.1, 0.2), (0.3, 4.3)]);
    }

    #[test]
    fn normalizing_with_0_1_does_nothing() {
        let front = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.2, 4.3],
            vec![0.6, 0.1, 1.3],
        ];
        let min_max = vec![(0.0, 1.0); 3];

        let normalized_front = normalize_front(&front, &min_max);

        assert_eq!(front, normalized_front);
    }

    #[test]
    fn normalizing_correctly() {
        let front = vec![
            vec![1.1, 1.2, 1.3],
            vec![1.4, 1.2, 4.3],
            vec![1.6, 1.1, 1.3],
            vec![1.0, 1.1, 1.0],
            vec![2.0, 2.1, 3.0],
        ];
        let min_max = vec![(1.0, 2.0), (1.1, 2.1), (1.0, 3.0)];

        let normalized_front = normalize_front(&front, &min_max);

        assert_eq!(
            normalized_front,
            vec![
                vec![
                    0.10000000000000009,
                    0.09999999999999987,
                    0.15000000000000002,
                ],
                vec![0.3999999999999999, 0.09999999999999987, 1.65],
                vec![0.6000000000000001, 0.0, 0.15000000000000002],
                vec![0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0],
            ]
        );
    }
}
