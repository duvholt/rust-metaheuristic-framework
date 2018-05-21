use itertools::Itertools;
use multiobjective::rank::calculate_fronts;
use solution::Solution;
use std::cmp::Ordering;
use std::f64::INFINITY;
use std::fmt::Debug;
use std::hash::Hash;

pub fn sort_on_objective(
    solutions: &Vec<&impl Solution<Vec<f64>>>,
    objective: usize,
) -> Vec<usize> {
    solutions
        .iter()
        .enumerate()
        .sorted_by(|(_, solution1), (_, solution2)| {
            let objectives = solution1.fitness().len();
            let mut order = Ordering::Equal;
            let mut objective = objective;
            while order == Ordering::Equal && objective < objectives {
                order = solution1.fitness()[objective]
                    .partial_cmp(&solution2.fitness()[objective])
                    .unwrap();
                objective += 1;
            }
            return order;
        })
        .into_iter()
        .map(|(i, _)| i)
        .collect()
}

pub fn crowding_distance(
    solutions: &Vec<&impl Solution<Vec<f64>>>,
    min: &Vec<f64>,
    max: &Vec<f64>,
) -> Vec<f64> {
    let mut distances = vec![0.0; solutions.len()];
    let objectives = solutions[0].fitness().len();
    for objective in 0..objectives {
        let sorted_indices = sort_on_objective(&solutions, objective);
        let first = *sorted_indices.first().unwrap();
        let last = *sorted_indices.last().unwrap();
        // Always keep boundaries
        distances[first] = INFINITY;
        distances[last] = INFINITY;
        // Skip boundaries
        for i in 1..sorted_indices.len() - 1 {
            let current = sorted_indices[i];
            let previous = &solutions[sorted_indices[i - 1]];
            let next = &solutions[sorted_indices[i + 1]];
            distances[current] += (next.fitness()[objective] - previous.fitness()[objective])
                / (max[objective] - min[objective])
        }
    }
    distances
}

fn crowding_comparison(
    (distance1, rank1): (f64, usize),
    (distance2, rank2): (f64, usize),
) -> Ordering {
    if rank1 < rank2 {
        Ordering::Less
    } else if rank1 == rank2 {
        if distance1 > distance2 {
            Ordering::Less
        } else if distance1 == distance2 {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    } else {
        Ordering::Greater
    }
}

fn sort_on_crowding<S>(solutions: Vec<S>, distances: Vec<f64>, ranks: Vec<usize>) -> Vec<(usize, S)>
where
    S: Solution<Vec<f64>>,
{
    izip!(solutions, distances, ranks)
        .enumerate()
        .sorted_by(|(_, (_, distance1, rank1)), (_, (_, distance2, rank2))| {
            crowding_comparison((*distance1, *rank1), (*distance2, *rank2))
        })
        .into_iter()
        .map(|(i, (solution, _, _))| (i, solution))
        .collect()
}

pub fn min_max_fitness(solutions: &[&impl Solution<Vec<f64>>]) -> (Vec<f64>, Vec<f64>) {
    let objectives = solutions[0].fitness().len();
    let mut min = vec![INFINITY; objectives];
    let mut max = vec![-INFINITY; objectives];
    for solution in solutions {
        for (objective, &fitness) in solution.fitness().into_iter().enumerate() {
            if min[objective] > fitness {
                min[objective] = fitness;
            }
            if max[objective] < fitness {
                max[objective] = fitness;
            }
        }
    }
    (min, max)
}

pub fn sort<S>(solutions: Vec<S>) -> Vec<(usize, S)>
where
    S: Solution<Vec<f64>> + Eq + Hash + Clone + Debug,
{
    let mut distances = vec![0.0; solutions.len()];
    let mut ranks = vec![0; solutions.len()];
    let (min_fitness, max_fitness) = min_max_fitness(&solutions.iter().collect::<Vec<_>>());
    let fronts = calculate_fronts(&solutions);
    for (rank, front) in fronts.into_iter().enumerate() {
        let front_solutions: Vec<_> = front.iter().map(|&i| &solutions[i]).collect();
        let front_distances = crowding_distance(&front_solutions, &min_fitness, &max_fitness);
        for (i, distance) in front.into_iter().zip(front_distances) {
            distances[i] = distance;
            ranks[i] = rank;
        }
    }
    sort_on_crowding(solutions, distances, ranks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use testing::solution::vec_to_multi_solution;

    #[test]
    fn test_sort_on_objective_first() {
        let solutions = vec_to_multi_solution(vec![
            vec![3.0, 6.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![1.5, 4.0],
            vec![4.0, 3.0],
        ]);

        let indicies = sort_on_objective(&solutions.iter().collect(), 0);

        assert_eq!(indicies, vec![1, 3, 2, 0, 4]);
    }

    #[test]
    fn test_sort_on_objective_second() {
        let solutions = vec_to_multi_solution(vec![
            vec![3.0, 6.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![1.5, 4.0],
            vec![4.0, 3.0],
        ]);

        let indicies = sort_on_objective(&solutions.iter().collect(), 1);

        assert_eq!(indicies, vec![2, 4, 3, 1, 0]);
    }

    #[test]
    fn test_min_max_fitness() {
        let solutions = vec_to_multi_solution(vec![
            vec![3.0, 6.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![1.5, 4.0],
            vec![4.0, 3.0],
            vec![0.5, 7.0],
            vec![5.0, 6.5],
        ]);

        let (min_fitness, max_fitness) = min_max_fitness(&solutions.iter().collect::<Vec<_>>());

        assert_eq!(min_fitness, vec![0.5, 2.0]);
        assert_eq!(max_fitness, vec![5.0, 7.0]);
    }

    #[test]
    fn test_crowding_distance() {
        let solutions = vec_to_multi_solution(vec![
            vec![3.0, 6.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![1.5, 4.0],
            vec![4.0, 3.0],
            vec![0.5, 7.0],
            vec![5.0, 6.5],
        ]);
        let (min_fitness, max_fitness) = min_max_fitness(&solutions.iter().collect::<Vec<_>>());

        let distances = crowding_distance(&solutions.iter().collect(), &min_fitness, &max_fitness);

        assert_eq!(
            distances,
            vec![
                0.7444444444444445,
                0.6222222222222222,
                INFINITY,
                0.6222222222222222,
                0.8444444444444444,
                INFINITY,
                INFINITY,
            ]
        );
    }

    #[test]
    fn test_non_dominated_sorting() {
        let solutions = vec_to_multi_solution(vec![
            vec![6.0, 1.0],  // 1, rank 0
            vec![6.0, 4.0],  // 2, rank 1
            vec![9.0, 5.0],  // 3, rank 2
            vec![5.0, 8.0],  // 4, rank 2
            vec![2.0, 7.0],  // 5, rank 1
            vec![2.0, 10.0], // 6, rank 2
            vec![4.0, 5.0],  // 7, rank 1
            vec![1.0, 10.0], // 8, rank 1
            vec![8.0, 3.0],  // 9, rank 1
            vec![1.0, 6.0],  // 10, rank 0
            vec![4.0, 2.0],  // 11, rank 0
            vec![2.0, 4.0],  // 12, rank 0
        ]);
        let clone_solutions = solutions.clone();

        let solutions = sort(solutions);

        let indices: Vec<_> = solutions
            .iter()
            .map(|(_, solution)| clone_solutions.iter().position(|s| solution == s).unwrap())
            .collect();
        assert_eq!(indices, vec![0, 9, 10, 11, 7, 8, 4, 6, 1, 2, 5, 3]);
    }

    #[ignore]
    #[bench]
    fn bench_non_dominated_sorting(b: &mut Bencher) {
        let solutions = vec_to_multi_solution(vec![
            vec![6.0, 1.0],  // 1, rank 0
            vec![6.0, 4.0],  // 2, rank 1
            vec![9.0, 5.0],  // 3, rank 2
            vec![5.0, 8.0],  // 4, rank 2
            vec![2.0, 7.0],  // 5, rank 1
            vec![2.0, 10.0], // 6, rank 2
            vec![4.0, 5.0],  // 7, rank 1
            vec![1.0, 10.0], // 8, rank 1
            vec![8.0, 3.0],  // 9, rank 1
            vec![1.0, 6.0],  // 10, rank 0
            vec![4.0, 2.0],  // 11, rank 0
            vec![2.0, 4.0],  // 12, rank 0
        ]);

        b.iter(|| sort(solutions.clone()));
    }
}
