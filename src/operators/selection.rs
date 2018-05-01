use multiobjective::non_dominated_sorting::crowding_distance;
use rand::{seq, weak_rng, Rng};
use solution::Solution;
use std::cmp::{min, Ordering};
use std::fmt::Debug;

pub fn roulette_wheel<S>(population: &[S]) -> (usize, &S)
where
    S: Solution<f64>,
{
    let mut rng = weak_rng();
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

pub fn tournament_selection<S>(
    population: &[S],
    tournament_size: usize,
    mut rng: impl Rng,
) -> (usize, &S)
where
    S: Solution<f64> + Debug,
{
    let selected =
        seq::sample_iter(&mut rng, population.iter().enumerate(), tournament_size).unwrap();
    selected
        .iter()
        .cloned()
        .min_by(|(_, s1): &(usize, &S), (_, s2): &(usize, &S)| {
            s1.fitness()
                .partial_cmp(&s2.fitness())
                .unwrap_or(Ordering::Equal)
        })
        .unwrap()
}

pub fn tournament_selection_crowding<S>(
    population: &[S],
    tournament_size: usize,
    mut rng: impl Rng,
    distances: &Vec<f64>,
) -> usize
where
    S: Solution<Vec<f64>> + Debug + Clone,
{
    let tournament_size = min(tournament_size, population.len());
    let selected =
        seq::sample_iter(&mut rng, population.iter().enumerate(), tournament_size).unwrap();
    let indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
    indices
        .into_iter()
        .max_by(|&i, &j| distances[i].partial_cmp(&distances[j]).unwrap())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, StdRng};

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

    fn create_rng() -> StdRng {
        let seed: &[_] = &[1, 2, 3, 4];
        SeedableRng::from_seed(seed)
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

    #[test]
    fn tournament_selection_selects_global_best() {
        let mut rng = create_rng();
        let population = vec![
            TestFitness::new(0.4),
            TestFitness::new(0.1),
            TestFitness::new(0.3),
            TestFitness::new(0.2),
        ];

        let (index, selected) = tournament_selection(&population, 4, &mut rng);

        assert_eq!(index, 1);
        assert_eq!(selected, &population[1]);
    }

    #[test]
    fn tournament_selection_selects_local_best() {
        let mut rng = create_rng();
        let population = vec![
            TestFitness::new(0.4),
            TestFitness::new(0.1),
            TestFitness::new(0.3),
            TestFitness::new(0.2),
        ];

        let (index, selected) = tournament_selection(&population, 2, &mut rng);

        assert_eq!(index, 3);
        assert_eq!(selected, &population[3]);
    }
}
