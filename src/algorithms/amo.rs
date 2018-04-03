use fitness_evaluation::FitnessEvaluator;
use rand;
use rand::distributions::normal::StandardNormal;
use rand::{thread_rng, Rng};
use solution::Solution;
use solution::SolutionJSON;

pub struct Config {
    pub iterations: i64,
    pub population: usize,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub dimension: usize,
    pub radius: usize,
}

#[derive(Clone)]
struct Animal {
    fitness: f64,
    position: Vec<f64>,
}

impl Solution<f64> for Animal {
    fn fitness(&self) -> &f64 {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

struct Herd<'a> {
    config: &'a Config,
    population: Vec<Animal>,
    fitness_evaluator: &'a FitnessEvaluator<'a, f64>,
}

impl<'a> Herd<'a> {
    fn new(config: &'a Config, fitness_evaluator: &'a FitnessEvaluator<f64>) -> Herd<'a> {
        Herd {
            config,
            population: vec![],
            fitness_evaluator,
        }
    }
    fn generate_next_generation(&self, population: &Vec<Animal>) -> Vec<Animal> {
        (0..population.len())
            .map(|i| {
                let mut new_animal = population[i].clone();
                for d in 0..self.config.dimension {
                    let mut rng = thread_rng();
                    let mut index_offset =
                        rng.gen_range(i - self.config.radius, i + self.config.radius) as i64;
                    let index = get_random_neighbor_index(index_offset, population.len());
                    let StandardNormal(gaussian) = rand::random();
                    new_animal.position[index] +=
                        gaussian * (population[index].position[d] - population[i].position[d]);
                }
                new_animal
            })
            .collect()
    }
}

pub fn get_random_neighbor_index(index_offset: i64, length: usize) -> usize {
    let index = index_offset % length as i64;
    if index < 0 {
        return (index + length as i64) as usize;
    }
    index as usize
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let mut solutions = vec![];
    let mut i = 1;
    while (i <= config.iterations) {
        i += 1;
    }
    solutions
}

#[cfg(test)]
mod tests {
    use super::*;
    use statistics::sampler::{Sampler, SamplerMode};
    use std::u64::MAX;
    use test::Bencher;
    use test_functions::rosenbrock;
    use testing::utils::{create_evaluator, create_sampler};

    fn create_config() -> Config {
        Config {
            upper_bound: 4.0,
            lower_bound: -4.0,
            dimension: 2,
            iterations: 100,
            population: 50,
            radius: 2,
        }
    }

    #[test]
    fn get_random_neighbor_index_test() {
        assert_eq!(get_random_neighbor_index(5, 50), 5);
        assert_eq!(get_random_neighbor_index(-2, 50), 48);
        assert_eq!(get_random_neighbor_index(52, 50), 2);
    }

}
