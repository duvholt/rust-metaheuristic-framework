use rand::{thread_rng, Rng};
use solution::Solution;

pub fn roulette_wheel<S>(population: &[S]) -> (usize, &S)
where
    S: Solution<f64>,
{
    let mut rng = thread_rng();
    let weight_sum: f64 = population.iter().map(|p| p.fitness()).sum();
    let mut threshold = rng.next_f64() * weight_sum;
    for (p_i, p) in population.iter().enumerate() {
        threshold -= p.fitness();
        if threshold < 0.0 {
            return (p_i, p);
        }
    }
    // Return last element if none was selected because of float arithmetic
    (population.len() - 1, population.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestFitness {
        fitness: f64,
        position: Vec<f64>,
    }

    impl TestFitness {
        fn new(fitness: f64) -> TestFitness {
            TestFitness {
                fitness,
                position: vec![],
            }
        }
    }

    impl PartialEq for TestFitness {
        fn eq(&self, other: &TestFitness) -> bool {
            self.fitness == other.fitness
        }
    }

    impl Solution<f64> for TestFitness {
        fn fitness(&self) -> &f64 {
            &self.fitness
        }

        fn position(&self) -> &Vec<f64> {
            &self.position
        }
    }

    #[test]
    fn roulette_wheel_selects_largest() {
        let fitness1 = TestFitness::new(0.0);
        let fitness2 = TestFitness::new(0.0);
        let fitness3 = TestFitness::new(1.0);
        let population = vec![fitness1.clone(), fitness2.clone(), fitness3.clone()];

        let (index, selected) = roulette_wheel(&population).clone();
        assert_eq!(index, 2);
        assert_eq!(selected, &fitness3);
    }

    #[test]
    fn roulette_wheel_selects_largest2() {
        let fitness1 = TestFitness::new(0.0);
        let fitness2 = TestFitness::new(1.0);
        let fitness3 = TestFitness::new(0.0);
        let population = vec![fitness1.clone(), fitness2.clone(), fitness3.clone()];

        let (index, selected) = roulette_wheel(&population).clone();
        assert_eq!(index, 1);
        assert_eq!(selected, &fitness2);
    }
}
