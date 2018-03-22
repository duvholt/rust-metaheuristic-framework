use solution::SolutionJSON;
use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::f64;
use solution::Solution;

pub struct Config {
    pub iterations: i64,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub dimensions: usize,
    pub r: f64,
    pub e: f64,
    pub min_seeds: i64,
    pub max_seeds: i64,
    pub normal_seeds: i64,
    pub self_learning_seeds: i64,
}

#[derive(Clone)]
struct Dandelion {
    core_dandelion: Seed,
    seeds: Vec<Seed>,
}

impl Solution for Dandelion {
    fn fitness(&self) -> f64 {
        self.core_dandelion.fitness
    }

    fn position(&self) -> Vec<f64> {
        self.core_dandelion.position.to_vec()
    }
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
            core_dandelion: Seed {
                fitness: self.calculate_fitness(&position),
                position,
            },
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
        for i in 0..self.population.len() {
            self.population[i].seeds = (0..self.config.normal_seeds)
                .map(|_| {
                    let mut position = self.population[i].core_dandelion.position.clone();
                    let distance = rng.gen_range(-radius, radius);
                    let index = (self.config.dimensions as f64 * rng.next_f64()) as usize;
                    if position[index] + distance > self.config.upper_bound
                        || (position[index] + distance) < self.config.lower_bound
                    {
                        position[index] =
                            rng.gen_range(self.config.lower_bound, self.config.upper_bound);
                    } else {
                        position[index] += distance;
                    }
                    let fitness = self.calculate_fitness(&position);
                    Seed { fitness, position }
                })
                .collect();
        }
    }

    fn self_learning_sowing(&mut self) {
        let mut rng = thread_rng();
        let index = (self.config.dimensions as f64 * rng.next_f64()) as usize;
        for i in 0..self.population.len() {
            let average_position = self.find_average_seed_position(&self.population[i]);
            let mut seeds: Vec<Seed> = (0..self.config.self_learning_seeds)
                .map(|_| {
                    let mut position = self.population[i].core_dandelion.position.clone();
                    let distance = position[index] - average_position[index];
                    if position[index] + distance > self.config.upper_bound
                        || position[index] + distance < self.config.lower_bound
                    {
                        position[index] =
                            rng.gen_range(self.config.lower_bound, self.config.upper_bound);
                    } else {
                        position[index] += distance;
                    }

                    Seed {
                        fitness: self.calculate_fitness(&position),
                        position,
                    }
                })
                .collect();
            self.population[i].seeds.append(&mut seeds);
        }
    }

    fn find_average_seed_position(&self, dandelion: &Dandelion) -> Vec<f64> {
        let mut position = vec![0.0; self.config.dimensions];
        for seed in &dandelion.seeds {
            for j in 0..self.config.dimensions {
                position[j] += seed.position[j];
            }
        }
        (0..position.len())
            .map(|i| position[i] / dandelion.seeds.len() as f64)
            .collect()
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
        sum += dandelion.core_dandelion.fitness;
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
            swarm.population[best_index].core_dandelion.fitness,
        );
        swarm.dandelion_sowing(radius);
        swarm.self_learning_sowing();
        let mut best: Seed = swarm.population[0].core_dandelion.clone();
        for i in 0..swarm.population.len() {
            for j in 0..swarm.population[i].seeds.len() {
                if swarm.population[i].seeds[j].fitness < best.fitness {
                    best = swarm.population[i].seeds[j].clone();
                }
            }
        }
        println!(
            "best fitness: {}, itteration{}, pos: {:?}",
            best.fitness, i, best.position
        );
        swarm.population[0].core_dandelion = best;
        swarm.population[0].seeds.clear();

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
            normal_seeds: 50,
            self_learning_seeds: 2,
        }
    }

    fn create_dandelion_with_fitness(fitness: f64) -> Dandelion {
        Dandelion {
            core_dandelion: Seed {
                position: vec![2.0, 1.0],
                fitness,
            },
            seeds: vec![],
        }
    }

    #[test]
    fn find_average_seed_position_test() {
        let config = create_config();
        let mut swarm = Swarm {
            test_function: &rosenbrock,
            config: &config,
            population: vec![],
        };
        let mut dandelion = create_dandelion_with_fitness(10.0);
        dandelion.seeds.push(Seed {
            fitness: 0.0,
            position: vec![2.0, 2.0],
        });
        assert_eq!(swarm.find_average_seed_position(&dandelion), vec![2.0, 2.0]);

        dandelion.seeds.push(Seed {
            fitness: 0.0,
            position: vec![4.0, 6.0],
        });

        assert_eq!(swarm.find_average_seed_position(&dandelion), vec![3.0, 4.0]);

        dandelion.seeds.push(Seed {
            fitness: 0.0,
            position: vec![-18.0, -8.0],
        });

        assert_eq!(
            swarm.find_average_seed_position(&dandelion),
            vec![-4.0, 0.0]
        );
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
