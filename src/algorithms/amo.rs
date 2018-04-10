use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use fitness_evaluation::FitnessEvaluator;
use position::random_position;
use rand::distributions::normal::StandardNormal;
use rand::{thread_rng, Rng};
use solution::Solution;
use solution::SolutionJSON;
use solution::sort_solutions_by_fitness;

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("Animal migration optimization")
        .arg(
            Arg::with_name("radius")
                .short("-r")
                .long("radius")
                .value_name("INTEGER")
                .help("neighborhood radius")
                .default_value("2")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<f64>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let radius = value_t_or_exit!(sub_m, "radius", i64);
    println!("Running AMO with radius: {}", radius);
    let config = Config {
        upper_bound: common.upper_bound,
        lower_bound: common.lower_bound,
        dimension: common.dimension,
        iterations: common.iterations,
        population: common.population,
        radius,
    };

    run(config, &function_evaluator)
}

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

impl PartialEq for Animal {
    fn eq(&self, other: &Animal) -> bool {
        self.fitness == other.fitness
    }
}

impl Solution<f64> for Animal {
    fn fitness(&self) -> &f64 {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

fn animal_migration(
    population: Vec<Animal>,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<f64>,
    config: &Config,
) -> Vec<Animal> {
    let moved_animals: Vec<Animal> = population
        .iter()
        .enumerate()
        .map(|(i, current_animal)| {
            let mut moved_position = vec![];
            let mut out_of_bounds = false;
            let StandardNormal(gaussian) = rng.gen();
            for d in 0..config.dimension {
                let mut index_offset =
                    rng.gen_range(i as i64 - config.radius, i as i64 + config.radius) as i64;
                let index = get_random_neighbor_index(index_offset, population.len());
                moved_position.push(
                    current_animal.position[d]
                        + gaussian * (population[index].position[d] - current_animal.position[d]),
                );
                if moved_position[d] > config.upper_bound || moved_position[d] < config.lower_bound
                {
                    out_of_bounds = true;
                    break;
                }
            }
            if out_of_bounds {
                moved_position =
                    random_position(config.lower_bound, config.upper_bound, config.dimension);
            }
            Animal {
                fitness: fitness_evaluator.calculate_fitness(&moved_position),
                position: moved_position,
            }
        })
        .collect();
    find_best_solutions(population, moved_animals)
}

fn animal_replacement(
    population: Vec<Animal>,
    best_animal: &Animal,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<f64>,
    config: &Config,
) -> Vec<Animal> {
    let new_population: Vec<Animal> = population
        .iter()
        .enumerate()
        .map(|(i, current_animal)| {
            let mut new_position = vec![];
            let mut fitness = current_animal.fitness;
            let mut changed = false;
            let mut out_of_bounds = false;
            for d in 0..config.dimension {
                let r = get_two_unique_numbers(population.len(), population.len(), &mut rng);
                if rng.next_f64() > (population.len() - i) as f64 / population.len() as f64 {
                    changed = true;
                    let r1 = population[r.0].position[d];
                    let r2 = population[r.1].position[d];
                    let best = best_animal.position[d];
                    let current = current_animal.position[d];
                    new_position.push(
                        r1 + rng.next_f64() * (best - current) + rng.next_f64() * (r2 - current),
                    );
                    if new_position[d] > config.upper_bound || new_position[d] < config.lower_bound
                    {
                        out_of_bounds = true;
                    }
                } else {
                    new_position.push(current_animal.position[d]);
                }
            }
            if changed {
                if out_of_bounds {
                    new_position =
                        random_position(config.lower_bound, config.upper_bound, config.dimension);
                }
                fitness = fitness_evaluator.calculate_fitness(&new_position);
            }
            Animal {
                position: new_position,
                fitness,
            }
        })
        .collect();
    find_best_solutions(population, new_population)
}

fn find_best_solutions(old_population: Vec<Animal>, new_population: Vec<Animal>) -> Vec<Animal> {
    old_population
        .into_iter()
        .zip(new_population)
        .map(
            |(old, new)| {
                if old.fitness > new.fitness {
                    new
                } else {
                    old
                }
            },
        )
        .collect()
}

fn get_random_neighbor_index(index_offset: i64, length: usize) -> usize {
    let index = index_offset % length as i64;
    if index < 0 {
        return (index + length as i64) as usize;
    }
    index as usize
}

fn get_two_unique_numbers(i: usize, length: usize, mut rng: impl Rng) -> (usize, usize) {
    let mut r1 = i;
    let mut r2 = i;
    while r1 == i {
        r1 = rng.gen_range(0, length);
    }
    while r2 == r1 || r2 == i {
        r2 = rng.gen_range(0, length);
    }
    (r1, r2)
}

fn generate_random_population(
    size: usize,
    fitness_evaluator: &FitnessEvaluator<f64>,
    config: &Config,
) -> Vec<Animal> {
    let mut population: Vec<Animal> = (0..size)
        .map(|_| generate_random_animal(&fitness_evaluator, &config))
        .collect();
    sort_solutions_by_fitness(&mut population);
    population
}

fn generate_random_animal(fitness_evaluator: &FitnessEvaluator<f64>, config: &Config) -> Animal {
    let position = random_position(config.lower_bound, config.upper_bound, config.dimension);
    Animal {
        fitness: fitness_evaluator.calculate_fitness(&position),
        position,
    }
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let solutions = vec![];
    let mut population: Vec<Animal> =
        generate_random_population(config.population, &fitness_evaluator, &config);
    let mut best_animal = population[0].clone();
    let mut rng = thread_rng();
    for i in 0..config.iterations {
        population = animal_migration(population, &mut rng, &fitness_evaluator, &config);
        sort_solutions_by_fitness(&mut population);
        population = animal_replacement(
            population,
            &best_animal,
            &mut rng,
            &fitness_evaluator,
            &config,
        );

        for animal in &population {
            if animal.fitness < best_animal.fitness {
                best_animal = animal.clone();
            }
        }
        fitness_evaluator
            .sampler
            .population_sample_single(i, &population);
        if fitness_evaluator.end_criteria() {
            break;
        }
    }
    fitness_evaluator
        .sampler
        .population_sample_single(config.iterations, &population);
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

    fn create_seedable_rng() -> StdRng {
        let seed: &[_] = &[1, 2, 3, 4];
        SeedableRng::from_seed(seed)
    }

    #[test]
    fn generate_random_population_test() {
        let config = create_config();
        let sampler = create_sampler();
        let fitness_evaluator = &create_evaluator(&sampler);
        let population = generate_random_population(5, &fitness_evaluator, &config);
        assert_eq!(population.len(), 5);
    }

    #[test]
    fn get_two_unique_numbers_test() {
        let i = 2;
        let length = 3;
        let mut rng = create_seedable_rng();
        assert_eq!(get_two_unique_numbers(i, length, &mut rng), (1, 0));
        assert_eq!(get_two_unique_numbers(i, length, &mut rng), (0, 1));
        assert_eq!(get_two_unique_numbers(i, length, &mut rng), (0, 1));
    }

    #[test]
    fn get_random_neighbor_index_test() {
        assert_eq!(get_random_neighbor_index(5, 50), 5);
        assert_eq!(get_random_neighbor_index(-2, 50), 48);
        assert_eq!(get_random_neighbor_index(52, 50), 2);
    }

    #[test]
    fn animal_replacement_test() {
        let config = create_config();
        let sampler = create_sampler();
        let fitness_evaluator = &create_evaluator(&sampler);
        let best = Animal {
            position: vec![0.1, 0.2],
            fitness: 0.3,
        };
        let population = vec![
            Animal {
                position: vec![1.0, 2.0],
                fitness: 3.0,
            },
            Animal {
                position: vec![2.0, 2.1],
                fitness: 4.1,
            },
            Animal {
                position: vec![3.0, 2.3],
                fitness: 5.3,
            },
            Animal {
                position: vec![4.0, 2.6],
                fitness: 6.6,
            },
        ];

        let mut rng = create_seedable_rng();
        let new_population =
            animal_replacement(population, &best, rng, &fitness_evaluator, &config);
        assert_eq!(new_population[0].fitness, 3.0);
        assert_eq!(new_population[1].fitness, 4.1);
        assert_eq!(new_population[2].fitness, 4.0448939501300325);
        assert_eq!(new_population[3].fitness, 5.388251274328821);
    }

    #[test]
    fn animal_migration_test() {
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

        let mut rng = create_seedable_rng();
        let next_generation = animal_migration(population, rng, &fitness_evaluator, &config);
        assert_eq!(next_generation[2].fitness, 5.0);
    }
}
