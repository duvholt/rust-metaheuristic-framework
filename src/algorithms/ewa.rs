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

    fn create_worm(&self, position: Vec<f64>) -> Worm {
        let fitness = self.calculate_fitness(&position);
        Worm { position, fitness }
    }

    fn generate_population(&self, size: i32) -> Vec<Worm> {
        (0..size)
            .map(|_| self.create_worm(self.random_position()))
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

    fn reproduction1(&self, worm: &Worm) -> Worm {
        let minmax = -self.config.space + self.config.space;
        let alpha = self.config.similarity;
        let mut new_position = vec![];
        for j in 0..self.config.dimension as usize {
            let x_j = minmax - alpha * worm.position[j];
            new_position.push(x_j);
        }
        let fitness = self.calculate_fitness(&new_position);
        Worm {
            position: new_position,
            fitness,
        }
    }

    fn combine_worms(&self, worm1: &Worm, worm2: &Worm, iteration: i64) -> Worm {
        let beta = 0.9f64.powf(iteration as f64) * self.config.beta;
        let mut new_position = vec![];
        for j in 0..self.config.dimension as usize {
            let x_j = beta * worm1.position[j] + (1.0 - beta) * worm2.position[j];
            new_position.push(x_j);
        }
        let fitness = self.calculate_fitness(&new_position);
        Worm {
            position: new_position,
            fitness,
        }
    }
}

pub fn run(config: Config, test_function: &Fn(&Vec<f64>) -> f64) -> Vec<Solution> {
    println!("Running Earth Worm Optimization. Example: {:?}", config);
    let mut worms = Worms::new(&config, &test_function);
    worms.population = worms.generate_population(config.population);
    for iteration in 0..config.iterations {
        worms.sort_population();
        let mut new_worms = vec![];
        for (worm_index, worm) in worms.population.iter().enumerate() {
            let offspring1 = worms.reproduction1(&worm);
            // TODO: Remove
            let offspring2 = worm.clone();
            if worm_index < config.n_kew {
                // Reproduction 2
            } else {
                // Select random from worms
            }
            let new_worm = worms.combine_worms(&offspring1, &offspring2, iteration);
            if worm_index < config.n_kew {
                // Cauchy mutation
            }
            new_worms.push(new_worm);
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

    #[test]
    fn sorts_population_by_ascending_fitness() {
        let config = create_config();
        let mut worms = Worms::new(&config, &rosenbrock);
        let dimension = config.dimension as usize;
        let worm1 = worms.create_worm(vec![0.3; dimension]);
        let worm2 = worms.create_worm(vec![0.2; dimension]);
        let worm3 = worms.create_worm(vec![0.02; dimension]);
        let worm4 = worms.create_worm(vec![0.4; dimension]);
        worms.population = vec![worm1.clone(), worm2.clone(), worm3.clone(), worm4.clone()];

        worms.sort_population();

        assert_eq!(worms.population, vec![worm3, worm2, worm1, worm4]);
    }

    #[test]
    fn reproduction1_generates_offspring() {
        let config = create_config();
        let mut worms = Worms::new(&config, &rosenbrock);
        let dimension = config.dimension as usize;
        let worm1 = worms.create_worm(vec![0.3; dimension]);

        let offspring = worms.reproduction1(&worm1);

        assert_eq!(
            offspring.position,
            vec![-0.3 * config.similarity; dimension]
        );
    }

    #[test]
    fn combines_worms_initial() {
        let config = create_config();
        let mut worms = Worms::new(&config, &rosenbrock);
        let dimension = config.dimension as usize;
        let worm1 = worms.create_worm(vec![1.0; dimension]);
        let worm2 = worms.create_worm(vec![2.0; dimension]);

        let combined = worms.combine_worms(&worm1, &worm2, 0);

        assert_eq!(combined.position, vec![1.0; dimension]);
    }

    #[test]
    fn combines_worms_iteration2() {
        let config = create_config();
        let mut worms = Worms::new(&config, &rosenbrock);
        let dimension = config.dimension as usize;
        let worm1 = worms.create_worm(vec![1.0; dimension]);
        let worm2 = worms.create_worm(vec![2.0; dimension]);

        let combined = worms.combine_worms(&worm1, &worm2, 2);

        assert_eq!(combined.position, vec![1.19; dimension]);
    }
}
