use domination::dominates;
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

    fn decrement(&self) {
        self.dominated_by.set(self.dominated_by.get() - 1);
    }

    fn dominated_by(&self) -> usize {
        self.dominated_by.get()
    }
}

fn calculate_domation_count<S>(solutions: &[S]) -> Vec<Domination>
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
            }
        }
    }
    dominations
}

pub fn calculate_fronts<'a, S>(solutions: &'a [S]) -> Vec<Vec<usize>>
where
    S: Solution<Vec<f64>> + Eq + Hash,
{
    let dominations = calculate_domation_count(&solutions);
    let mut current_front = vec![];
    for (i, domination) in dominations.iter().enumerate() {
        if domination.dominated_by() == 0 {
            current_front.push((i, domination));
        }
    }
    let mut fronts = vec![];
    while current_front.len() > 0 {
        let mut new_front = vec![];
        fronts.push(current_front.iter().map(|(i, _)| *i).collect());
        for (_, domination) in current_front {
            for j in domination.dominates.iter().cloned() {
                dominations[j].decrement();
                if dominations[j].dominated_by() == 0 {
                    let other = &dominations[j];
                    new_front.push((j, other));
                }
            }
        }
        current_front = new_front;
    }
    fronts
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::solution::vec_to_multi_solution;

    #[test]
    fn test_increment_decrement_rank() {
        let mut domination = Domination::new();

        assert_eq!(domination.dominated_by(), 0);
        domination.increment();
        assert_eq!(domination.dominated_by(), 1);
        domination.decrement();
        assert_eq!(domination.dominated_by(), 0);
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
            .map(|domination| domination.dominated_by())
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

        let fronts = calculate_fronts(&solutions);

        let fronts_hashset = fronts
            .into_iter()
            .map(|front| front.into_iter().collect())
            .collect::<Vec<HashSet<_>>>();
        assert_eq!(
            fronts_hashset,
            vec![
                vec![0, 9, 10, 11].into_iter().collect(),
                vec![7, 8, 6, 1, 4].into_iter().collect(),
                vec![2, 5, 3].into_iter().collect(),
            ]
        );
    }
}