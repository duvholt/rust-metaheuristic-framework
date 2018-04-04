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
    pub radius: i64,
}

#[derive(Clone, Debug)]
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

fn generate_next_generation(
    population: Vec<Animal>,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<f64>,
    config: &Config,
) -> Vec<Animal> {
    let new_animals: Vec<Animal> = (0..population.len())
        .map(|i| {
            let mut new_animal = population[i].clone();
            for d in 0..config.dimension {
                let mut index_offset =
                    rng.gen_range(i as i64 - config.radius, i as i64 + config.radius) as i64;
                let index = get_random_neighbor_index(index_offset, population.len());
                let StandardNormal(gaussian) = rng.gen();
                new_animal.position[d] +=
                    gaussian * (population[index].position[d] - population[i].position[d]);
            }
            new_animal.fitness = fitness_evaluator.calculate_fitness(&new_animal.position);
            new_animal
        })
        .collect();
    population
        .into_iter()
        .zip(new_animals)
        .map(|(old, new)| {
            println!("old {:?}    new {:?}", old, new);
            if old.fitness > new.fitness {
                new
            } else {
                old
            }
        })
        .collect()
}

fn get_random_neighbor_index(index_offset: i64, length: usize) -> usize {
    let index = index_offset % length as i64;
    if index < 0 {
        return (index + length as i64) as usize;
    }
    index as usize
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let mut solutions = vec![];
    let mut population: Vec<Animal> = vec![];
    let mut i = 1;

    while (i <= config.iterations) {
        i += 1;
    }
    solutions
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, StdRng};
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

    #[test]
    fn generate_next_generation_test() {
        let config = create_config();
        let sampler = create_sampler();
        let fitness_evaluator = &create_evaluator(&sampler);
        let population = vec![
            Animal {
                position: vec![1.0, 2.0],
                fitness: 1.0,
            },
            Animal {
                position: vec![2.0, 2.1],
                fitness: 10.0,
            },
            Animal {
                position: vec![3.0, 2.3],
                fitness: 5.0,
            },
            Animal {
                position: vec![4.0, 2.6],
                fitness: 1000.0,
            },
        ];

        let seed: &[_] = &[1, 2, 3, 4];
        let mut rng: StdRng = SeedableRng::from_seed(seed);
        let next_generation =
            generate_next_generation(population, rng, &fitness_evaluator, &config);
        assert_eq!(next_generation[0].fitness, 1.0);
        assert_eq!(next_generation[1].fitness, 2.584863028248279);
        assert_eq!(next_generation[2].fitness, 5.0);
        assert_eq!(next_generation[3].fitness, 5.4520381489022505);
    }
}
