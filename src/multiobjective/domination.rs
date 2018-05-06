use rand::Rng;
use solution::Solution;
use std::collections::HashSet;

pub fn dominates(a: &Vec<f64>, b: &Vec<f64>) -> bool {
    dominates_objectives(a, b, a.len())
}

// Dominates specified number of objectives
pub fn dominates_objectives(a: &Vec<f64>, b: &Vec<f64>, objectives: usize) -> bool {
    let mut equal = true;
    for i in 0..objectives {
        if a[i] > b[i] {
            return false;
        } else if a[i] < b[i] {
            equal = false;
        }
    }
    return !equal;
}

pub fn dominates_inverted_objectives(a: &Vec<f64>, b: &Vec<f64>, objectives: usize) -> bool {
    let mut equal = true;
    for i in 0..objectives {
        if a[i] < b[i] {
            return false;
        } else if a[i] > b[i] {
            equal = false;
        }
    }
    return !equal;
}

pub fn select_first(a: &Vec<f64>, b: &Vec<f64>, rng: &mut impl Rng) -> bool {
    if dominates(&a, &b) {
        return true;
    } else if dominates(b, a) {
        return false;
    } else {
        return rng.gen();
    }
}

pub fn find_non_dominated<M>(solutions: &[M]) -> HashSet<usize>
where
    M: Solution<Vec<f64>>,
{
    find_non_dominated_n_objectives(solutions, solutions[0].fitness().len(), false)
}

pub fn find_non_dominated_n_objectives<M>(
    solutions: &[M],
    objectives: usize,
    inverted: bool,
) -> HashSet<usize>
where
    M: Solution<Vec<f64>>,
{
    let mut non_dominated = HashSet::new();
    let dominates_func = if inverted {
        dominates_inverted_objectives
    } else {
        dominates_objectives
    };
    for (p_i, p) in solutions.iter().enumerate() {
        let mut dominated = false;
        non_dominated.retain(|&q_i| {
            let q: &M = &solutions[q_i];
            if &p.fitness()[0..objectives] == &q.fitness()[0..objectives] {
                return false;
            } else if dominates_func(&p.fitness(), &q.fitness(), objectives) {
                return false;
            } else if !dominated && dominates_func(&q.fitness(), &p.fitness(), objectives) {
                dominated = true;
            }
            return true;
        });
        if !dominated {
            non_dominated.insert(p_i);
        }
    }
    non_dominated
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use testing::utils::create_rng;

    #[test]
    fn dominates_all() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.2, 0.3, 0.4];

        let a_dominates_b = dominates(&a, &b);

        assert!(a_dominates_b);
    }

    #[test]
    fn dominates_one() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.3, 0.3];

        let a_dominates_b = dominates(&a, &b);

        assert!(a_dominates_b);
    }

    #[test]
    fn dominates_equal() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.2, 0.3];

        let a_dominates_b = dominates(&a, &b);

        assert!(!a_dominates_b);
    }

    #[test]
    fn dominates_equal2() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];

        let a_dominates_b = dominates(&a, &b);

        assert!(!a_dominates_b);
    }

    #[test]
    fn does_not_dominate() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.3, 0.1, 0.3];

        let a_dominates_b = dominates(&a, &b);

        assert!(!a_dominates_b);
    }

    #[derive(Clone)]
    struct TestMultiSolution {
        fitness: Vec<f64>,
    }

    impl Solution<Vec<f64>> for TestMultiSolution {
        fn fitness(&self) -> &Vec<f64> {
            &self.fitness
        }

        fn position(&self) -> &Vec<f64> {
            &self.fitness
        }
    }

    fn vec_to_multi_solution(solutions: Vec<Vec<f64>>) -> Vec<TestMultiSolution> {
        solutions
            .iter()
            .map(|solution| TestMultiSolution {
                fitness: solution.to_vec(),
            })
            .collect()
    }

    #[test]
    fn finds_non_dominated() {
        let solutions = vec_to_multi_solution(vec![
            vec![3.0, 4.0], // 0, dominated
            vec![1.0, 5.0], // 1, non-dominated
            vec![2.0, 2.0], // 2, non-dominated
            vec![1.5, 4.0], // 3, non-dominated
            vec![3.0, 3.0], // 4, dominated
            vec![4.0, 1.5], // 5, non-dominated
            vec![4.0, 5.0], // 6, dominated
        ]);

        let non_dominated_indexes = find_non_dominated(&solutions);

        let expected: HashSet<_> = [1, 2, 3, 5].iter().cloned().collect();
        assert_eq!(non_dominated_indexes, expected);
    }

    #[test]
    fn finds_non_dominated_equal() {
        let solutions = vec_to_multi_solution(vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]]);

        let non_dominated_indexes = find_non_dominated(&solutions);

        assert_eq!(non_dominated_indexes.len(), 1);
    }

    #[test]
    fn selects_first_when_dominating() {
        let fitness1 = vec![0.3, 0.2, 0.6];
        let fitness2 = vec![0.4, 0.2, 0.7];
        let mut rng = create_rng();

        let result = select_first(&fitness1, &fitness2, &mut rng);

        assert!(result);
    }

    #[test]
    fn selects_second_when_dominated() {
        let fitness1 = vec![0.3, 0.2, 0.6];
        let fitness2 = vec![0.4, 0.2, 0.7];
        let mut rng = create_rng();

        let result = select_first(&fitness1, &fitness2, &mut rng);

        assert!(result);
    }

    #[test]
    fn selects_random_when_non_dominated() {
        let fitness1 = vec![0.3, 0.2, 0.6];
        let fitness2 = vec![0.2, 0.2, 0.7];
        let mut rng = create_rng();

        let result = select_first(&fitness1, &fitness2, &mut rng);

        assert!(!result);
    }

    #[ignore]
    #[bench]
    fn bench_non_dominated(b: &mut Bencher) {
        let solutions = vec_to_multi_solution(vec![
            vec![3.0, 4.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![1.5, 4.0],
            vec![3.0, 3.0],
            vec![4.0, 1.5],
            vec![4.0, 5.0],
        ]);

        b.iter(|| find_non_dominated(&solutions))
    }
}
