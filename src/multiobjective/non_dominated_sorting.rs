use solution::Solution;
use std::f64::INFINITY;
use itertools::Itertools;

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

#[cfg(test)]
mod tests {
    use super::*;
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
}