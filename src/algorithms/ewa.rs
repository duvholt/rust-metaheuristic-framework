use solution::Solution;
use rand::distributions::{IndependentSample, Range};
use rand::thread_rng;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct Config {
    space: f64,
    dimension: i32,
    iterations: i64,
    population: i32,
    n_kew: usize,
    beta: f64,
    similarity: f64,
}

#[derive(Clone, Debug)]
struct Worm {
    position: Vec<f64>,
    fitness: f64,
}

impl PartialEq for Worm {
    fn eq(&self, other: &Worm) -> bool {
        self.position == other.position && self.fitness == other.fitness
    }
}

struct Worms<'a> {
    config: &'a Config,
    population: Vec<Worm>,
    test_function: &'a Fn(&Vec<f64>) -> f64,
}

impl<'a> Worms<'a> {
    fn new(config: &'a Config, test_function: &'a Fn(&Vec<f64>) -> f64) -> Worms<'a> {
        Worms {
            config,
            population: vec![],
            test_function,
        }
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        (self.test_function)(x)
    }

    fn random_position(&self) -> Vec<f64> {
        let between = Range::new(-self.config.space, self.config.space);
        let mut rng = thread_rng();
        (0..self.config.dimension)
            .map(|_| between.ind_sample(&mut rng))
            .collect()
    }

    fn generate_population(&self, size: i32) -> Vec<Worm> {
        (0..size)
            .map(|_| {
                let position = self.random_position();
                let fitness = self.calculate_fitness(&position);
                Worm {
                    position: position.to_vec(),
                    fitness,
                }
            })
            .collect()
    }

    pub fn solutions(&self) -> Vec<Solution> {
        let mut solutions: Vec<Solution> = self.population
            .iter()
            .map(|worm| Solution {
                x: worm.position.to_vec(),
                fitness: worm.fitness,
            })
            .collect();
        solutions
            .sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
        solutions
    }

    fn sort_population(&mut self) {
        self.population
            .sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
    }
}

pub fn run(config: Config, test_function: &Fn(&Vec<f64>) -> f64) -> Vec<Solution> {
    println!("Running Earth Worm Optimization. Example: {:?}", config);
    let mut worms = Worms::new(&config, &test_function);
    worms.population = worms.generate_population(config.population);
    for _ in 0..config.iterations {
        worms.sort_population();
        let mut new_worms = vec![];
        for (worm_index, _worm) in worms.population.iter().enumerate() {
            // Reproduction 1
            if worm_index < config.n_kew {
                // Reproduction 2
            } else {
                // Select random from worms
            }
            // Combine
            if worm_index < config.n_kew {
                // Cauchy mutation
            }
        }
        worms.population = new_worms;
    }
    worms.solutions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions::rosenbrock;

    fn create_config() -> Config {
        Config {
            space: 4.0,
            dimension: 2,
            iterations: 50,
            population: 50,
            n_kew: 50,
            beta: 1.0,
            similarity: 0.98,
        }
    }

    fn create_worm(fitness: f64, dimension: i32) -> Worm {
        Worm {
            position: vec![fitness; dimension as usize],
            fitness,
        }
    }

    #[test]
    fn sorts_population_by_ascending_fitness() {
        let config = create_config();
        let mut worms = Worms::new(&config, &rosenbrock);
        let worm1 = create_worm(0.3, config.dimension);
        let worm2 = create_worm(0.2, config.dimension);
        let worm3 = create_worm(0.02, config.dimension);
        let worm4 = create_worm(0.4, config.dimension);
        worms.population = vec![worm1.clone(), worm2.clone(), worm3.clone(), worm4.clone()];

        worms.sort_population();

        assert_eq!(worms.population, vec![worm3, worm2, worm1, worm4]);
    }
}
