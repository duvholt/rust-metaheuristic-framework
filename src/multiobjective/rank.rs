use domination::dominates;
use solution::Solution;
use std::collections::HashSet;
use std::hash::Hash;

#[derive(Clone)]
struct Domination<'a, S: 'a>
where
    S: Solution<Vec<f64>> + Eq + Hash,
{
    dominates: HashSet<(usize)>,
    solution: &'a S,
    dominated_by: usize,
}

impl<'a, S: Solution<Vec<f64>> + Eq + Hash> Domination<'a, S> {
    fn new(solution: &'a S) -> Domination<'a, S> {
        Domination {
            dominates: HashSet::new(),
            solution,
            dominated_by: 0,
        }
    }

    fn add(&mut self, solution: usize) {
        self.dominates.insert(solution);
    }

    fn increment(&mut self) {
        self.dominated_by += 1;
    }

    fn decrement(&mut self) {
        self.dominated_by -= 1;
    }
}

fn calculate_domation_count<'a, S>(solutions: &'a [S]) -> Vec<Domination<'a, S>>
where
    S: Solution<Vec<f64>> + Eq + Hash,
{
    let mut dominations: Vec<_> = solutions
        .iter()
        .map(|solution| Domination::new(solution))
        .collect();
    for (i, solution1) in solutions.iter().enumerate() {
        for (j, solution2) in solutions.iter().enumerate() {
            if i == j {
                continue;
            }
            if dominates(&solution1.fitness(), &solution2.fitness()) {
                dominations[i].add(j);
                dominations[j].increment();
            }
        }
    }
    dominations
}

pub fn calculate_ranks<'a, S>(solutions: &'a [S]) -> Vec<usize>
where
    S: Solution<Vec<f64>> + Eq + Hash + Clone,
{
    let mut dominations = calculate_domation_count(&solutions);
    let mut ranks = vec![13123; solutions.len()];
    let mut front = vec![];
    for (i, domination) in dominations.iter().enumerate() {
        if domination.dominated_by == 0 {
            front.push(domination.clone());
            ranks[i] = 0;
        }
    }
    let mut front_count = 1;
    while front.len() > 0 {
        let mut new_front = vec![];
        for domination in front {
            for j in domination.dominates {
                dominations[j].decrement();
                if dominations[j].dominated_by == 0 {
                    let other = &dominations[j];
                    new_front.push(other.clone());
                    ranks[j] = front_count;
                }
            }
        }
        front_count += 1;
        front = new_front;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use solution::MultiTestSolution;
    use testing::solution::vec_to_multi_solution;

    #[test]
    fn test_increment_decrement_rank() {
        let solution = MultiTestSolution::new(vec![0.5]);
        let mut domination = Domination::new(&solution);

        assert_eq!(domination.dominated_by, 0);
        domination.increment();
        assert_eq!(domination.dominated_by, 1);
        domination.decrement();
        assert_eq!(domination.dominated_by, 0);
    }

    #[test]
    fn test_calculates_domination_count() {
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

        let dominations = calculate_domation_count(&solutions);

        let domination_count: Vec<_> = dominations
            .iter()
            .map(|domination| domination.dominated_by)
            .collect();
        assert_eq!(domination_count, vec![0, 3, 6, 5, 2, 4, 2, 1, 2, 0, 0, 0]);
    }

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

        let ranks = calculate_ranks(&solutions);

        assert_eq!(ranks, vec![0, 1, 2, 2, 1, 2, 1, 1, 1, 0, 0, 0]);
    }
}
