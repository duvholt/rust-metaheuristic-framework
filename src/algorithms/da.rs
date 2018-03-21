#[cfg(windows)]
#[link(name = "msvcrt")]
extern "C" {
    fn tgamma(x: f64) -> f64;
}

#[cfg(not(windows))]
#[link(name = "m")]
extern "C" {
    fn tgamma(x: f64) -> f64;
}
fn gamma(x: f64) -> f64 {
    unsafe { tgamma(x) }
}

use solution::Solution;
use rand::distributions::{IndependentSample, Range, Sample};
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::f64;

pub struct Config {
    pub iterations: i64,
    pub space: f64,
    pub dimensions: i32,
    pub r: f64,
    pub e: f64,
    pub min_seeds: i64,
    pub max_seeds: i64,
    pub seeds: i64
}

#[derive(Clone)]
struct Dandelion {
    fitness: f64,
    position: Vec<f64>,
    seeds: Vec<Seed>,
}

#[derive(Clone)]
struct Seed {
    fitness: f64,
    position: Vec<f64>,
}

struct Swarm<'a> {
    config: &'a Config,
    population: Vec<Dandelion>,
    test_function: &'a Fn(&Vec<f64>) -> f64,
}

impl<'a> Swarm<'a> {
    fn new(config: &'a Config, test_function: &'a Fn(&Vec<f64>) -> f64) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            test_function,
        }
    }

    fn generate_random_dandelion(&self) -> Dandelion {
        let between = Range::new(-self.config.space, self.config.space);
        let mut rng = thread_rng();
        let position = (0..self.config.dimensions)
            .map(|_| between.ind_sample(&mut rng))
            .collect();
        Dandelion {
            fitness: self.calculate_fitness(&position),
            position,
            seeds: vec![],

        }
    }
    
    fn generate_random_population(&self, size: i64) -> Vec<Dandelion> {
        let mut population = vec![];
        for _ in 0..size {
            population.push(self.generate_random_dandelion());
        }
        population
    }

    fn dandelion_sowing(&mut self, radius: f64) {
        let mut rng = thread_rng();
        let index = (self.config.dimensions as f64 * rng.next_f64()) as usize;
        for i in 0..self.population.len() {
            for _ in 0..self.config.seeds {
                let mut position = self.population[i].position.clone();
                position[index] = rng.next_f64() * radius;
                let seed = Seed {
                    fitness: self.calculate_fitness(&position),
                    position,  
                };
               self.population[i].seeds.push(seed);
            }
        }
    }

    fn calculate_number_of_seeds(
        &self,
        current_solution_fitness: f64,
        max_fitness: f64,
        min_fitness: f64,
    ) -> i64 {
        let mut seeds = (self.config.max_seeds as f64
            * ((max_fitness - current_solution_fitness + self.config.e)
                / (max_fitness - min_fitness))) as i64;
        if seeds > self.config.min_seeds {
            return seeds;
        } else {
            return self.config.min_seeds;
        }
    }

    fn calculate_assistant_radius(
        &self,
        current_iteration: i64,
        prev_radius: f64,
        current_dandelion: &Dandelion,
        core_dandelion: &Dandelion,
    ) -> f64 {
        if current_iteration == 1 {
            return self.config.space * 2.0;
        } else {
            let w = current_iteration as f64 / self.config.iterations as f64;
            let core_best = find_vector_max_value(&core_dandelion.position);
            let current_best = find_vector_max_value(&current_dandelion.position);
            return w * prev_radius + (core_best - current_best);
        }
    }

    fn calculate_core_radius(
        &self,
        current_iteration: i64,
        prev_radius: f64,
        prev_fitness: f64,
        current_fitness: f64,
    ) -> f64 {
        if current_iteration == 1 {
            return self.config.space * 2.0;
        } else {
            if current_fitness == prev_fitness {
                return prev_radius * self.config.r;
            } else {
                return prev_radius * self.config.e;
            }
        }
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        (self.test_function)(x)
    }
}

fn find_average_fitness(population: &Vec<Dandelion>, ) -> f64 {
    let mut sum = 0.0;
    for dandelion in population {
        sum += dandelion.fitness;
    }
    sum / population.len() as f64
}

fn find_vector_max_value(vec: &Vec<f64>) -> f64 {
    vec.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .cloned()
        .unwrap()
}

fn levy_flight(beta: f64) {
    let numerator = gamma(1.0 + beta) * (f64::consts::PI / 2.0).sin();
    let denominator = gamma((1.0 + beta) / 2.0) * beta * 2.0_f64.powf((beta - 1.0) / 2.0);
    let sigma = (numerator / denominator).powf(1.0 / beta);

    let mut between = Range::new(0.0, sigma.powi(2));
    let mut rng = thread_rng();
    let u = between.sample(&mut rng);
    let v = rng.next_f64();

    (u / v.abs().powf(1.0 / beta));
}

pub fn run(config: Config, test_function: &Fn(&Vec<f64>) -> f64) -> Vec<Solution> {
    let popuation_size = 1;
    let mutation_popuation_size = 1;
    let mut solutions = vec![];

    let mut i = 1;
    let mut rng = thread_rng();
    let mut swarm = Swarm::new(&config, &test_function);
    let mut prev_radius = 0.0;
    let mut prev_fitness = 0.0;
    let best_index = 0;

    swarm.population = swarm.generate_random_population(popuation_size);
    while i <= config.iterations {
        let radius = swarm.calculate_core_radius(i, prev_radius, prev_fitness, swarm.population[best_index].fitness);
        swarm.dandelion_sowing(radius);  
        i += 1;
    }
    solutions
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions::rosenbrock;
    use test::Bencher;

    fn create_config() -> Config {
        Config {
            space: 4.0,
            dimensions: 2,
            iterations: 100,
            r: 0.95,
            e: 1.05,
            min_seeds: 10,
            max_seeds: 100,
            seeds: 50,
        }
    }

    fn create_dandelion_with_fitness(fitness: f64) -> Dandelion {
        Dandelion {
            position: vec![2.0, 1.0],
            fitness,
            seeds: vec![],
        }
}

    #[test]
    fn find_vector_max_value_test() {
        let vector = vec![2.0, 3.9, 4.4];
        let max_value = find_vector_max_value(&vector);
        assert_eq!(max_value, 4.4);
    }

    #[test]
    fn calculate_assistant_radius_test() {
        let config = create_config();
        let mut swarm = Swarm {
            test_function: &rosenbrock,
            config: &config,
            population: vec![],
        };
        let mut core = Dandelion {
            fitness: 2.0,
            position: vec![1.0],
            seeds: vec![],
        };
        let mut current = Dandelion {
            fitness: 1.0,
            position: vec![2.0],
            seeds: vec![],
        };

        assert_eq!(
            swarm.calculate_assistant_radius(1, 2.0, &current, &core),
            8.0
        );
        assert_eq!(
            swarm.calculate_assistant_radius(2, 2.0, &current, &core),
            -0.96
        );
        assert_approx_eq!(
            swarm.calculate_assistant_radius(23, 3.4, &current, &core),
            -0.218
        );
        current.position.push(3.0);
        core.position.push(1.1);
        assert_approx_eq!(
            swarm.calculate_assistant_radius(23, 3.4, &current, &core),
            -1.118
        );
    }

    #[test]
    fn calculate_core_radius_test() {
        let config = create_config();
        let mut swarm = Swarm {
            test_function: &rosenbrock,
            config: &config,
            population: vec![],
        };

        assert_eq!(swarm.calculate_core_radius(1, 100.0, 10.0, 1000.0), 8.0);
        assert_eq!(swarm.calculate_core_radius(2, 2.1, 43.5, 51.1), 2.205);
        assert_approx_eq!(swarm.calculate_core_radius(2, 2.1, 43.5, 43.5), 1.995);
    }

    #[test]
    fn calculate_number_of_seeds_test() {
        let config = create_config();
        let mut swarm = Swarm {
            test_function: &rosenbrock,
            config: &config,
            population: vec![],
        };

        assert_eq!(swarm.calculate_number_of_seeds(50.0, 100.0, 10.0), 56);
        assert_eq!(swarm.calculate_number_of_seeds(95.9, 100.0, 10.0), 10);
    }

    #[test]
    fn gamma_function_test() {
        assert_eq!(4.0 * 3.0 * 2.0 * 1.0, gamma(5.0));
    }

    #[test]
    fn find_average_fitness_test() {
        let population = vec![
            create_dandelion_with_fitness(5.0),
            create_dandelion_with_fitness(10.0),
            create_dandelion_with_fitness(15.0),
        ];
        let average = find_average_fitness(&population);

        assert_eq!(average, 10.0);
    }

    #[bench]
    fn bench_da(b: &mut Bencher) {
        b.iter(|| {
            let config = create_config();
            run(config, &rosenbrock);
        });
    }
}
