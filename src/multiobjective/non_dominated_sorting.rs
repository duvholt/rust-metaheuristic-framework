use itertools::Itertools;
use multiobjective::rank::calculate_ranks;
use solution::Solution;
use std::cmp::Ordering;
use std::f64::INFINITY;
use std::fmt::Debug;
use std::hash::Hash;

fn sort_on_objective(solutions: &[impl Solution<Vec<f64>>], objective: usize) -> Vec<usize> {
    solutions
        .iter()
        .enumerate()
        .map(|(i, solution)| (i, solution.fitness()[objective]))
        .sorted_by(|(_, fitness1), (_, fitness2)| fitness1.partial_cmp(&fitness2).unwrap())
        .into_iter()
        .map(|(i, _)| i)
        .collect()
}

pub fn crowding_distance(solutions: &[impl Solution<Vec<f64>>]) -> Vec<f64> {
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
            distances[current] += next.fitness()[objective] - previous.fitness()[objective]
        }
    }
    distances
}

#[derive(Debug)]
struct DominatedSort<S>
where
    S: Solution<Vec<f64>> + Debug,
{
    solution: S,
    rank: usize,
    distance: f64,
}

pub fn sort<S>(solutions: Vec<S>) -> Vec<S>
where
    S: Solution<Vec<f64>> + Eq + Hash + Clone + Debug,
{
    let distances = crowding_distance(&solutions);
    let ranks = calculate_ranks(&solutions);

    let mut dominated_sort: Vec<_> = solutions
        .into_iter()
        .zip(distances)
        .zip(ranks)
        .map(|((solution, distance), rank)| DominatedSort {
            solution,
            rank,
            distance,
        })
        .collect();
    println!("{:?}", dominated_sort);
    dominated_sort.sort_unstable_by(|d1, d2| {
        if d1.rank < d2.rank {
            Ordering::Less
        } else if d1.rank == d2.rank {
            if d1.distance > d2.distance {
                Ordering::Less
            } else if d1.distance == d2.distance {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        } else {
            Ordering::Greater
        }
    });
    dominated_sort
        .into_iter()
        .map(|d| d.solution.clone())
        .collect()
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

        let indicies = sort_on_objective(&solutions, 0);

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

        let indicies = sort_on_objective(&solutions, 1);

        assert_eq!(indicies, vec![2, 4, 3, 1, 0]);
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

        let distances = crowding_distance(&solutions);

        assert_eq!(
            distances,
            vec![3.5, 3.0, INFINITY, 3.0, 4.0, INFINITY, INFINITY]
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
            .map(|solution| clone_solutions.iter().position(|s| solution == s).unwrap())
            .collect();
        assert_eq!(indices, vec![0, 9, 10, 11, 7, 8, 1, 4, 6, 2, 3, 5]);
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
