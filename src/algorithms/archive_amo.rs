use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use fitness_evaluation::FitnessEvaluator;
use multiobjective::crowding_archive::Archive;
use multiobjective::domination::dominates;
use multiobjective::domination::select_first;
use multiobjective::rank::calculate_domation_count;
use operators::mutation;
use operators::position::multi_random_position;
use rand::distributions::normal::StandardNormal;
use rand::{weak_rng, Rng};
use solution::{Solution, SolutionJSON};
use std::hash;

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("Archive Animal Migration Optimization")
        .arg(
            Arg::with_name("radius")
                .short("-r")
                .long("radius")
                .value_name("INTEGER")
                .help("neighborhood radius")
                .default_value("2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("archive_size")
                .short("-a")
                .long("archive_size")
                .value_name("archive_size")
                .help("archive size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("divisions")
                .long("divisions")
                .value_name("divisions")
                .help("number of archive divisions")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<Vec<f64>>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let radius = value_t_or_exit!(sub_m, "radius", i64);
    let archive_size = value_t!(sub_m, "archive_size", usize).unwrap_or(common.population);
    let divisions = value_t!(sub_m, "divisions", usize).unwrap_or(30);
    if common.verbose >= 1 {
        println!("Running Archive-AMO with radius: {}", radius);
    }
    let config = Config {
        multi_upper_bound: common.multi_upper_bound.clone(),
        multi_lower_bound: common.multi_lower_bound.clone(),
        dimensions: common.dimensions,
        iterations: common.iterations,
        population: common.population,
        radius,
        archive_size,
        divisions,
    };

    run(config, &function_evaluator)
}

pub struct Config {
    pub iterations: i64,
    pub population: usize,
    pub multi_upper_bound: Vec<f64>,
    pub multi_lower_bound: Vec<f64>,
    pub dimensions: usize,
    pub radius: i64,
    pub archive_size: usize,
    pub divisions: usize,
}

#[derive(Clone, Debug)]
struct Animal {
    fitness: Vec<f64>,
    position: Vec<f64>,
}

impl hash::Hash for Animal {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.position_to_notnan().hash(state)
    }
}

impl PartialEq for Animal {
    fn eq(&self, other: &Animal) -> bool {
        self.position == other.position
    }
}

impl Eq for Animal {}

impl Solution<Vec<f64>> for Animal {
    fn fitness(&self) -> &Vec<f64> {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

fn animal_migration(
    population: Vec<Animal>,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<Vec<f64>>,
    config: &Config,
) -> Vec<Animal> {
    let moved_animals: Vec<Animal> = population
        .iter()
        .enumerate()
        .map(|(i, current_animal)| {
            let mut moved_position = Vec::with_capacity(config.dimensions);
            let StandardNormal(gaussian) = rng.gen();
            for d in 0..config.dimensions {
                let mut index_offset =
                    rng.gen_range(i as i64 - config.radius, i as i64 + config.radius) as i64;
                let index = get_random_neighbor_index(index_offset, population.len());
                let mut pos_d = current_animal.position[d]
                    + gaussian * (population[index].position[d] - current_animal.position[d]);
                if pos_d > config.multi_upper_bound[d] {
                    pos_d = config.multi_upper_bound[d];
                } else if pos_d < config.multi_lower_bound[d] {
                    pos_d = config.multi_lower_bound[d];
                }
                moved_position.push(pos_d);
            }
            Animal {
                fitness: fitness_evaluator.calculate_fitness(&moved_position),
                position: moved_position,
            }
        })
        .collect();
    find_best_solutions(population, moved_animals, rng)
}

fn animal_replacement(
    population: Vec<Animal>,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<Vec<f64>>,
    config: &Config,
    archive: &Archive<Animal>,
) -> Vec<Animal> {
    let new_population: Vec<Animal> = {
        let best_animal = archive.select_leader();
        let probabilities = find_probabilities(&population);
        population
            .iter()
            .enumerate()
            .map(|(i, current_animal)| {
                let mut new_position = Vec::with_capacity(config.dimensions);
                let mut fitness = current_animal.fitness.clone();
                let mut changed = false;
                for d in 0..config.dimensions {
                    if rng.next_f64() > probabilities[i] {
                        changed = true;
                        let r = get_two_unique_numbers(i, population.len(), &mut rng);
                        let r1 = population[r.0].position[d];
                        let r2 = population[r.1].position[d];
                        let best = best_animal.position[d];
                        let current = current_animal.position[d];
                        let mut pos_d = r1
                            + rng.next_f64() * (best - current)
                            + rng.next_f64() * (r2 - current);
                        if pos_d > config.multi_upper_bound[d] {
                            pos_d = config.multi_upper_bound[d];
                        } else if pos_d < config.multi_lower_bound[d] {
                            pos_d = config.multi_lower_bound[d];
                        }
                        new_position.push(pos_d);
                    } else {
                        new_position.push(current_animal.position[d]);
                    }
                }
                if changed {
                    fitness = fitness_evaluator.calculate_fitness(&new_position);
                }
                Animal {
                    position: new_position,
                    fitness,
                }
            })
            .collect()
    };
    find_best_solutions(population, new_population, rng)
}

fn find_probabilities(solutions: &Vec<Animal>) -> Vec<f64> {
    let dominations = calculate_domation_count(&solutions);
    let mut probabilities = vec![0.0; solutions.len()];
    for (i, domination) in dominations.into_iter().enumerate() {
        probabilities[i] =
            (solutions.len() - domination.dominated_by()) as f64 / solutions.len() as f64;
    }
    probabilities
}

fn find_best_solutions(
    old_population: Vec<Animal>,
    new_population: Vec<Animal>,
    mut rng: impl Rng,
) -> Vec<Animal> {
    old_population
        .into_iter()
        .zip(new_population)
        .map(|(old, new)| {
            if dominates(&old.fitness, &new.fitness) {
                old
            } else if dominates(&new.fitness, &old.fitness) {
                new
            } else {
                // If neither dominates the other select randomly
                let r = rng.next_f64();
                if r > 0.5 {
                    old
                } else {
                    new
                }
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
    fitness_evaluator: &FitnessEvaluator<Vec<f64>>,
    config: &Config,
) -> Vec<Animal> {
    (0..size)
        .map(|_| generate_random_animal(&fitness_evaluator, &config))
        .collect()
}

fn generate_random_animal(
    fitness_evaluator: &FitnessEvaluator<Vec<f64>>,
    config: &Config,
) -> Animal {
    let position = multi_random_position(
        &config.multi_lower_bound,
        &config.multi_upper_bound,
        config.dimensions,
    );
    Animal {
        fitness: fitness_evaluator.calculate_fitness(&position),
        position,
    }
}

fn mutate_population(
    population: Vec<Animal>,
    iteration: i64,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<Vec<f64>>,
    config: &Config,
) -> Vec<Animal> {
    population
        .into_iter()
        .map(|animal| {
            let mutated_position = mutation::one_dimension(
                &mut rng,
                &animal.position,
                &config.multi_lower_bound,
                &config.multi_upper_bound,
                iteration,
                config.iterations,
                0.1,
            );
            if let Some(position) = mutated_position {
                let fitness = fitness_evaluator.calculate_fitness(&position);
                if select_first(&animal.fitness, &fitness, &mut rng) {
                    animal
                } else {
                    Animal { position, fitness }
                }
            } else {
                animal
            }
        })
        .collect()
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<Vec<f64>>) -> Vec<SolutionJSON> {
    let mut archive = Archive::new(config.archive_size);
    let mut population: Vec<Animal> =
        generate_random_population(config.population, &fitness_evaluator, &config);
    let mut rng = weak_rng();
    archive.update(&population);
    for i in 0..config.iterations {
        population = animal_migration(population, &mut rng, &fitness_evaluator, &config);
        archive.update(&population);
        if fitness_evaluator.end_criteria() {
            break;
        }
        population = mutate_population(population, i, &mut rng, fitness_evaluator, &config);
        if fitness_evaluator.end_criteria() {
            break;
        }
        population =
            animal_replacement(population, &mut rng, &fitness_evaluator, &config, &archive);
        archive.update(&population);
        fitness_evaluator
            .sampler
            .population_sample_multi(i, &archive.get_population());
        if fitness_evaluator.end_criteria() {
            break;
        }
    }
    fitness_evaluator
        .sampler
        .population_sample_multi(config.iterations, &archive.get_population());
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, StdRng};
    use test::Bencher;
    use testing::utils::{create_evaluator_multi, create_sampler_multi};

    fn create_config() -> Config {
        Config {
            multi_upper_bound: vec![4.0; 2],
            multi_lower_bound: vec![-4.0; 2],
            dimensions: 2,
            iterations: 100,
            population: 50,
            radius: 2,
            archive_size: 50,
            divisions: 10,
        }
    }

    fn create_seedable_rng() -> StdRng {
        let seed: &[_] = &[1, 2, 3, 4];
        SeedableRng::from_seed(seed)
    }

    #[test]
    fn generate_random_population_test() {
        let config = create_config();
        let sampler = create_sampler_multi();
        let fitness_evaluator = &create_evaluator_multi(&sampler);
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
        let sampler = create_sampler_multi();
        let fitness_evaluator = &create_evaluator_multi(&sampler);
        let mut archive = Archive::new(config.archive_size);

        let population = vec![
            Animal {
                position: vec![0.1, 0.2],
                fitness: vec![0.1, 0.2],
            },
            Animal {
                position: vec![2.0, 2.1],
                fitness: vec![2.0, 2.1],
            },
            Animal {
                position: vec![3.0, 2.3],
                fitness: vec![3.0, 2.3],
            },
            Animal {
                position: vec![4.0, 2.6],
                fitness: vec![4.0, 2.6],
            },
        ];
        archive.update(&population);
        let rng = create_seedable_rng();
        let new_population =
            animal_replacement(population, rng, &fitness_evaluator, &config, &archive);
        assert_eq!(new_population[0].fitness, vec![0.1, 0.2]);
        assert_eq!(new_population[1].fitness, vec![2.0, 2.1]);
        assert_eq!(new_population[2].fitness, vec![1.3356651356791132, 2.3]);
        assert_eq!(
            new_population[3].fitness,
            vec![-1.306348201401438, 1.5661140513601106]
        );
    }

    #[test]
    fn animal_migration_test() {
        let config = create_config();
        let sampler = create_sampler_multi();
        let fitness_evaluator = &create_evaluator_multi(&sampler);
        let mut archive = Archive::new(config.archive_size);

        let population = vec![
            Animal {
                position: vec![1.0, 2.0],
                fitness: vec![1.0, 2.0],
            },
            Animal {
                position: vec![2.0, 2.1],
                fitness: vec![2.0, 2.1],
            },
            Animal {
                position: vec![3.0, 2.3],
                fitness: vec![3.0, 2.3],
            },
            Animal {
                position: vec![4.0, 2.6],
                fitness: vec![4.0, 2.6],
            },
        ];
        archive.update(&population);
        let rng = create_seedable_rng();
        let next_generation = animal_migration(population, rng, &fitness_evaluator, &config);
        assert_eq!(next_generation[2].fitness, vec![3.0, 2.0433037454089833]);
    }

    #[ignore]
    #[bench]
    fn bench_amo(b: &mut Bencher) {
        b.iter(|| {
            let sampler = create_sampler_multi();
            let evaluator = create_evaluator_multi(&sampler);
            let config = create_config();
            run(config, &evaluator)
        });
    }
}
