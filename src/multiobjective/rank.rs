use multiobjective::domination::dominates;
use multiobjective::non_dominated_sorting::sort_on_objective;
use solution::Solution;
use std::cell::Cell;
use std::collections::HashSet;
use std::hash::Hash;

#[derive(Clone)]
pub struct Domination {
    dominates: HashSet<(usize)>,
    dominated_by: Cell<usize>,
}

impl Domination {
    fn new() -> Domination {
        Domination {
            dominates: HashSet::new(),
            dominated_by: Cell::new(0),
        }
    }

    fn add(&mut self, solution: usize) {
        self.dominates.insert(solution);
    }

    fn increment(&mut self) {
        self.dominated_by.set(self.dominated_by.get() + 1);
    }

    pub fn dominated_by(&self) -> usize {
        self.dominated_by.get()
    }

    fn dominates(&self, solution: &usize) -> bool {
        self.dominates.contains(solution)
    }
}

pub fn calculate_domation_count<S>(solutions: &[S]) -> Vec<Domination>
where
    S: Solution<Vec<f64>> + Eq + Hash,
{
    let mut dominations: Vec<_> = solutions.iter().map(|_| Domination::new()).collect();
    for (i, solution1) in solutions.iter().enumerate() {
        for (j, solution2) in solutions.iter().enumerate() {
            if i == j {
                continue;
            }
            if dominates(&solution1.fitness(), &solution2.fitness()) {
                dominations[i].add(j);
                dominations[j].increment();
            } else if &solution1.fitness() == &solution2.fitness() {
                if !dominations[j].dominates(&i) {
                    dominations[i].add(j);
                    dominations[j].increment();
                }
            }
        }
    }
    dominations
}

fn sequential_search_front<S>(solution: &S, fronts: &Vec<Vec<usize>>, solutions: &[S]) -> usize
where
    S: Solution<Vec<f64>>,
{
    'outer: for k in 0..fronts.len() {
        for &solution_index in fronts[k].iter().rev() {
            let fitness1 = solutions[solution_index].fitness();
            let fitness2 = solution.fitness();
            if fitness1 == fitness2 || dominates(fitness1, fitness2) {
                continue 'outer;
            }
        }
        return k;
    }
    return fronts.len();
}

pub fn calculate_fronts<'a, S>(solutions: &'a [S]) -> Vec<Vec<usize>>
where
    S: Solution<Vec<f64>> + Clone,
{
    let mut fronts = vec![];
    let sorted_indices = sort_on_objective(&solutions.iter().collect(), 0);
    for sorted_index in sorted_indices {
        let front_index = sequential_search_front(&solutions[sorted_index], &fronts, &solutions);
        if front_index >= fronts.len() {
            fronts.push(vec![sorted_index]);
        } else {
            fronts[front_index].push(sorted_index);
        }
    }
    fronts
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::solution::vec_to_multi_solution;
    #[test]
    fn test_calculates_ranks() {
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

        let fronts = calculate_fronts(&solutions);

        assert_eq!(
            fronts,
            vec![vec![9, 11, 10, 0], vec![7, 4, 6, 1, 8], vec![5, 3, 2]]
        );
    }

    #[test]
    fn test_calculates_ranks_with_equal_fitness() {
        let solutions = vec_to_multi_solution(vec![
            vec![6.0, 1.0], // 1, rank 0
            vec![6.0, 1.0], // 2, rank 1
        ]);

        let fronts = calculate_fronts(&solutions);

        assert_eq!(fronts, vec![vec![0], vec![1]]);
    }
}
