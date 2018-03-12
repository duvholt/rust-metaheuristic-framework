use rand::{thread_rng, Rng};

pub trait Fitness {
    fn fitness(&self) -> f64;
}

pub fn roulette_wheel<F>(population: &[F]) -> (usize, &F)
where
    F: Fitness,
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
    }

    impl PartialEq for TestFitness {
        fn eq(&self, other: &TestFitness) -> bool {
            self.fitness == other.fitness
        }
    }

    impl Fitness for TestFitness {
        fn fitness(&self) -> f64 {
            self.fitness
        }
    }

    #[test]
    fn roulette_wheel_selects_largest() {
        let fitness1 = TestFitness { fitness: 0.0 };
        let fitness2 = TestFitness { fitness: 0.0 };
        let fitness3 = TestFitness { fitness: 1.0 };
        let population = vec![fitness1.clone(), fitness2.clone(), fitness3.clone()];

        let (index, selected) = roulette_wheel(&population).clone();
        assert_eq!(index, 2);
        assert_eq!(selected, fitness3);
    }

    #[test]
    fn roulette_wheel_selects_largest2() {
        let fitness1 = TestFitness { fitness: 0.0 };
        let fitness2 = TestFitness { fitness: 1.0 };
        let fitness3 = TestFitness { fitness: 0.0 };
        let population = vec![fitness1.clone(), fitness2.clone(), fitness3.clone()];

        let (index, selected) = roulette_wheel(&population).clone();
        assert_eq!(index, 1);
        assert_eq!(selected, fitness2);
    }
}
