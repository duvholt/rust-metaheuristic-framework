use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use fitness_evaluation::FitnessEvaluator;
use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng};
use solution::Solution;
use solution::SolutionJSON;
use std::f64;

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("Dandelion algorithm")
        .arg(
            Arg::with_name("r")
                .short("-r")
                .long("r")
                .value_name("r")
                .help("r constant")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("e")
                .short("-e")
                .long("e")
                .value_name("e")
                .help("e constant")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("normal_seeds")
                .short("-n")
                .long("normal_seeds")
                .value_name("normal_seeds")
                .help("normal_seeds constant")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("self_learning_seeds")
                .short("-s")
                .long("self_learning_seeds")
                .value_name("self_learning_seeds")
                .help("self_learning_seeds constant")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<f64>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let r = value_t!(sub_m, "r", f64).unwrap_or(0.95);
    let e = value_t!(sub_m, "e", f64).unwrap_or(1.05);
    let normal_seeds = value_t!(sub_m, "normal_seeds", i64).unwrap_or(200);
    let self_learning_seeds = value_t!(sub_m, "self_learning_seeds", i64).unwrap_or(10);
    println!(
        "Running DA with r: {} e: {} normal_seeds: {} self_learning_seeds: {}",
        r, e, normal_seeds, self_learning_seeds
    );
    let config = Config {
        upper_bound: common.upper_bound,
        lower_bound: common.lower_bound,
        dimension: common.dimension,
        iterations: common.iterations,
        population: common.population,
        r,
        e,
        normal_seeds,
        self_learning_seeds,
    };

    run(config, &function_evaluator)
}

pub struct Config {
    pub iterations: i64,
    pub population: usize,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub dimension: usize,
    pub r: f64,
    pub e: f64,
    pub normal_seeds: i64,
    pub self_learning_seeds: i64,
}

#[derive(Clone)]
struct Dandelion {
    core_dandelion: Seed,
    seeds: Vec<Seed>,
    radius: f64,
    old_fitness: f64,
}

#[derive(Clone)]
struct Seed {
    fitness: f64,
    position: Vec<f64>,
}

impl Solution<f64> for Seed {
    fn fitness(&self) -> &f64 {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

struct Swarm<'a> {
    config: &'a Config,
    population: Vec<Dandelion>,
    fitness_evaluator: &'a FitnessEvaluator<'a, f64>,
}

impl<'a> Swarm<'a> {
    fn new(config: &'a Config, fitness_evaluator: &'a FitnessEvaluator<f64>) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            fitness_evaluator,
        }
    }

    fn generate_random_dandelion(&self) -> Dandelion {
        let between = Range::new(self.config.lower_bound, self.config.upper_bound);
        let mut rng = thread_rng();
        let position = (0..self.config.dimension)
            .map(|_| between.ind_sample(&mut rng))
            .collect();
        Dandelion {
            core_dandelion: Seed {
                fitness: self.calculate_fitness(&position),
                position,
            },
            seeds: vec![],
            radius: 1.0,
            old_fitness: f64::INFINITY,
        }
    }

    fn generate_random_population(&self, size: usize) -> Vec<Dandelion> {
        let mut population = vec![];
        for _ in 0..size {
            population.push(self.generate_random_dandelion());
        }
        population
    }

    fn dandelion_sowing(&mut self) {
        let mut rng = thread_rng();
        for i in 0..self.population.len() {
            self.population[i].seeds = (0..self.config.normal_seeds)
                .map(|_| {
                    let mut position = self.population[i].core_dandelion.position.clone();
                    let radius = self.population[i].radius;
                    let distance = rng.gen_range(-radius, radius);
                    let index = (self.config.dimension as f64 * rng.next_f64()) as usize;
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
        for i in 0..self.population.len() {
            let average_position = self.find_average_seed_position(&self.population[i]);
            let mut seeds: Vec<Seed> = (0..self.config.self_learning_seeds)
                .map(|_| {
                    let mut position = self.population[i].core_dandelion.position.clone();
                    let index = (self.config.dimension as f64 * rng.next_f64()) as usize;
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
        let mut position = vec![0.0; self.config.dimension];
        for seed in &dandelion.seeds {
            for j in 0..self.config.dimension {
                position[j] += seed.position[j];
            }
        }
        (0..position.len())
            .map(|i| position[i] / dandelion.seeds.len() as f64)
            .collect()
    }

    fn calculate_sowing_radius(&mut self, current_iteration: i64) {
        for dandelion in &mut self.population {
            if current_iteration == 1 {
                dandelion.radius = self.config.upper_bound - self.config.lower_bound;
            } else {
                if dandelion.core_dandelion.fitness == dandelion.old_fitness {
                    dandelion.radius *= self.config.r;
                } else {
                    dandelion.radius *= self.config.e;
                }
            }
        }
    }

    fn select_best_seed(&mut self) {
        for i in 0..self.population.len() {
            let mut best: Seed = self.population[i].core_dandelion.clone();
            for j in 0..self.population[i].seeds.len() {
                if self.population[i].seeds[j].fitness < best.fitness {
                    best = self.population[i].seeds[j].clone();
                }
            }
            let old_fitness = self.population[i].core_dandelion.fitness;
            self.population[i].core_dandelion = best;
            self.population[i].old_fitness = old_fitness;
            self.population[i].seeds.clear();
            println!(
                "best fitness: {}, Dandelion number: {}, pos: {:?}",
                self.population[i].core_dandelion.fitness,
                i,
                self.population[i].core_dandelion.position
            );
        }
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        self.fitness_evaluator.calculate_fitness(x)
    }

    fn get_population(&self) -> Vec<Seed> {
        self.population
            .iter()
            .map(|dandelion| dandelion.core_dandelion.clone())
            .collect()
    }
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let mut i = 1;
    let mut swarm = Swarm::new(&config, &fitness_evaluator);
    let solutions = vec![];

    swarm.population = swarm.generate_random_population(config.population);
    while i < config.iterations {
        swarm.calculate_sowing_radius(i);
        swarm.dandelion_sowing();
        swarm.self_learning_sowing();
        swarm.select_best_seed();

        fitness_evaluator
            .sampler
            .population_sample_single(i, &swarm.get_population());
        if fitness_evaluator.end_criteria() {
            break;
        }
        i += 1;
    }
    fitness_evaluator
        .sampler
        .population_sample_single(config.iterations, &swarm.get_population());
    solutions
}

#[cfg(test)]
mod tests {
    use super::*;
    use statistics::sampler::{Sampler, SamplerMode};
    use test::Bencher;
    use test_functions::rosenbrock;

    fn create_sampler() -> Sampler {
        Sampler::new(10, 10, SamplerMode::Evolution)
    }

    fn create_evaluator(sampler: &Sampler) -> FitnessEvaluator<f64> {
        FitnessEvaluator::new(rosenbrock, 100, &sampler)
    }

    fn create_config() -> Config {
        Config {
            upper_bound: 4.0,
            lower_bound: -4.0,
            dimension: 2,
            iterations: 100,
            population: 1,
            r: 0.95,
            e: 1.05,
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
            radius: 1.0,
            old_fitness: f64::INFINITY,
        }
    }

    #[test]
    fn find_average_seed_position_test() {
        let config = create_config();
        let sampler = create_sampler();
        let swarm = Swarm {
            fitness_evaluator: &create_evaluator(&sampler),
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
    fn calculate_core_radius_test() {
        let config = create_config();
        let sampler = create_sampler();
        let mut swarm = Swarm {
            fitness_evaluator: &create_evaluator(&sampler),
            config: &config,
            population: vec![],
        };
        swarm.population.push(Dandelion {
            core_dandelion: Seed {
                position: vec![],
                fitness: 51.1,
            },
            seeds: vec![],
            radius: 2.1,
            old_fitness: 43.5,
        });
        swarm.calculate_sowing_radius(1);
        assert_eq!(swarm.population[0].radius, 8.0);

        swarm.population[0].radius = 2.1;
        swarm.calculate_sowing_radius(2);
        assert_eq!(swarm.population[0].radius, 2.205);

        swarm.population[0].radius = 2.1;
        swarm.population[0].core_dandelion.fitness = 43.5;
        swarm.calculate_sowing_radius(2);
        assert_approx_eq!(swarm.population[0].radius, 1.995);
    }

    #[ignore]
    #[bench]
    fn bench_da(b: &mut Bencher) {
        b.iter(|| {
            let sampler = create_sampler();
            let evaluator = create_evaluator(&sampler);
            let config = create_config();
            run(config, &evaluator);
        });
    }
}
