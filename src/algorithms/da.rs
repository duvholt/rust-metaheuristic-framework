use solution::SolutionJSON;
use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::f64;

pub struct Config {
    pub iterations: i64,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub dimensions: i32,
    pub r: f64,
    pub e: f64,
    pub min_seeds: i64,
    pub max_seeds: i64,
    pub seeds: i64,
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
        let between = Range::new(self.config.lower_bound, self.config.upper_bound);
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
            self.population[i].seeds = (0..self.config.seeds)
                .map(|_| {
                    let mut position = self.population[i].position.clone();
                    let distance = rng.gen_range(-radius, radius);

                    if (position[index] + radius) > self.config.upper_bound
                        || (position[index] + radius) < self.config.lower_bound
                    {
                        position[index] =
                            rng.gen_range(self.config.lower_bound, self.config.upper_bound)
                    } else {
                        position[index] += distance;
                    }
                    Seed {
                        fitness: self.calculate_fitness(&position),
                        position,
                    }
                })
                .collect();
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
            return self.config.upper_bound - self.config.lower_bound;
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

fn find_average_fitness(population: &Vec<Dandelion>) -> f64 {
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

pub fn run(config: Config, test_function: &Fn(&Vec<f64>) -> f64) -> Vec<SolutionJSON> {
    let popuation_size = 1;
    let mutation_popuation_size = 1;
    let mut solutions = vec![];

    let mut i = 1;
    let mut rng = thread_rng();
    let mut swarm = Swarm::new(&config, &test_function);
    let mut prev_radius = 1.0;
    let mut prev_fitness = 1.0;
    let best_index = 0;

    swarm.population = swarm.generate_random_population(popuation_size);
    while i <= config.iterations {
        let radius = swarm.calculate_core_radius(
            i,
            prev_radius,
            prev_fitness,
            swarm.population[best_index].fitness,
        );
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
            upper_bound: 4.0,
            lower_bound: -4.0,
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
