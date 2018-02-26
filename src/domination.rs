use std::collections::HashSet;

pub fn dominates(a: &Vec<f64>, b: &Vec<f64>) -> bool {
    let mut equal = true;
    for i in 0..a.len() {
        if a[i] > b[i] {
            return false;
        } else if a[i] < b[i] {
            equal = false;
        }
    }
    return !equal;
}

pub fn find_non_dominated(solutions: &Vec<Vec<f64>>) -> HashSet<usize> {
    let mut non_dominated = HashSet::new();
    for (p_i, p) in solutions.iter().enumerate() {
        let mut dominated = false;
        non_dominated.retain(|&q_i| {
            let q = &solutions[q_i];
            if dominates(&p, &q) {
                return false;
            } else if dominates(&q, &p) {
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
    fn does_not_dominate() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.3, 0.1, 0.3];

        let a_dominates_b = dominates(&a, &b);

        assert!(!a_dominates_b);
    }

    #[test]
    fn finds_non_dominated() {
        let solutions = vec![
            vec![3.0, 4.0], // 0, dominated
            vec![1.0, 5.0], // 1, non-dominated
            vec![2.0, 2.0], // 2, non-dominated
            vec![1.5, 4.0], // 3, non-dominated
            vec![3.0, 3.0], // 4, dominated
            vec![4.0, 1.5], // 5, non-dominated
            vec![4.0, 5.0], // 6, dominated
        ];

        let non_dominated_indexes = find_non_dominated(&solutions);

        let expected: HashSet<_> = [1, 2, 3, 5].iter().cloned().collect();
        assert_eq!(non_dominated_indexes, expected);
    }

    #[bench]
    fn bench_non_dominated(b: &mut Bencher) {
        let solutions = vec![
            vec![3.0, 4.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![1.5, 4.0],
            vec![3.0, 3.0],
            vec![4.0, 1.5],
            vec![4.0, 5.0],
        ];

        b.iter(|| find_non_dominated(&solutions))
    }
}
