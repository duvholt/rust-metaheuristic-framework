use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use distribution::cauchy;
use fitness_evaluation::FitnessEvaluator;
use operators::position::{limit_position_random, random_position};
use operators::selection::roulette_wheel_minimize;
use rand::{weak_rng, Rng};
use solution::{solutions_to_json, Solution, SolutionJSON};
use std::f64::EPSILON;

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("earth worm optimization algorithm")
        .arg(
            Arg::with_name("beta")
                .short("b")
                .long("beta")
                .value_name("FLOAT")
                .help("beta constant")
                .default_value("1.0")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("similarity")
                .short("s")
                .long("similarity")
                .value_name("FLOAT")
                .help("similarity constant")
                .default_value("0.98")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("cooling_factor")
                .short("c")
                .long("cooling-factor")
                .value_name("FLOAT")
                .help("beta cooling factor")
                .default_value("0.9")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<f64>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let beta = value_t_or_exit!(sub_m, "beta", f64);
    let similarity = value_t_or_exit!(sub_m, "similarity", f64);
    let cooling_factor = value_t_or_exit!(sub_m, "cooling_factor", f64);
    if common.verbose >= 1 {
        println!(
            "Running EWA with beta: {} similarity: {}, cooling factor: {}",
            beta, similarity, cooling_factor
        );
    }

    let config = Config {
        upper_bound: common.upper_bound,
        lower_bound: common.lower_bound,
        dimensions: common.dimensions,
        iterations: common.iterations,
        population: common.population,
        beta,
        similarity,
        cooling_factor,
    };
    run(config, &function_evaluator)
}

#[derive(Debug)]
pub struct Config {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub dimensions: usize,
    pub iterations: i64,
    pub population: usize,
    pub beta: f64,
    pub similarity: f64,
    pub cooling_factor: f64,
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

impl Solution<f64> for Worm {
    fn fitness(&self) -> &f64 {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

struct Worms<'a> {
    config: &'a Config,
    population: Vec<Worm>,
    fitness_evaluator: &'a FitnessEvaluator<'a, f64>,
}

impl<'a> Worms<'a> {
    fn new(config: &'a Config, fitness_evaluator: &'a FitnessEvaluator<f64>) -> Worms<'a> {
        Worms {
            config,
            population: vec![],
            fitness_evaluator,
        }
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        self.fitness_evaluator.calculate_fitness(x)
    }

    fn random_position(&self) -> Vec<f64> {
        random_position(
            self.config.lower_bound,
            self.config.upper_bound,
            self.config.dimensions,
        )
    }

    fn create_worm(&self, position: Vec<f64>) -> Worm {
        let fitness = self.calculate_fitness(&position);
        Worm { position, fitness }
    }

    fn generate_population(&self, size: usize) -> Vec<Worm> {
        (0..size)
            .map(|_| self.create_worm(self.random_position()))
            .collect()
    }

    fn sort_population(&mut self) {
        self.population
            .sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    }

    fn reproduction1(&self, worm: &Worm) -> Vec<f64> {
        let minmax = self.config.lower_bound + self.config.upper_bound;
        let alpha = self.config.similarity;
        let mut new_position = vec![];
        for j in 0..self.config.dimensions {
            let x_j = minmax - alpha * worm.position[j];
            new_position.push(x_j);
        }
        limit_position_random(
            &mut new_position,
            self.config.lower_bound,
            self.config.upper_bound,
        );
        new_position
    }

    fn reproduction2(&self) -> Vec<f64> {
        let (parent1index, parent1) = roulette_wheel_minimize(&self.population);
        let parent2 = self.random_other_worm(parent1index);

        // uniform crossover
        let mut pos1 = vec![];
        let mut pos2 = vec![];
        let mut rng = weak_rng();
        for j in 0..self.config.dimensions {
            let r = rng.next_f64();
            let p1j = parent1.position[j];
            let p2j = parent2.position[j];
            if r > 0.5 {
                pos1.push(p1j);
                pos2.push(p2j);
            } else {
                pos1.push(p2j);
                pos2.push(p1j);
            }
        }

        let f1 = self.calculate_fitness(&pos1);
        let f2 = self.calculate_fitness(&pos2);

        let w1 = f2 / (f1 + f2 + EPSILON);
        let w2 = f1 / (f1 + f2 + EPSILON);

        let mut position = vec![];
        for j in 0..self.config.dimensions {
            position.push(w1 * pos1[j] + w2 * pos2[j]);
        }

        limit_position_random(
            &mut position,
            self.config.lower_bound,
            self.config.upper_bound,
        );
        position
    }

    fn combine_worms(&self, position1: &Vec<f64>, position2: &Vec<f64>, iteration: i64) -> Worm {
        let beta = self.config.cooling_factor.powf(iteration as f64) * self.config.beta;
        let mut new_position = (0..self.config.dimensions)
            .map(|j| beta * position1[j] + (1.0 - beta) * position2[j])
            .collect();
        limit_position_random(
            &mut new_position,
            self.config.lower_bound,
            self.config.upper_bound,
        );
        let fitness = self.calculate_fitness(&new_position);
        Worm {
            position: new_position,
            fitness,
        }
    }

    fn random_other_worm(&self, worm_index: usize) -> Worm {
        let mut rng = weak_rng();
        let mut other_index = worm_index;
        while other_index == worm_index {
            other_index = rng.gen_range(0, self.config.population);
        }
        self.population[other_index].clone()
    }

    fn cauchy_mutation(&self, worm: &Worm) -> Worm {
        let population_size = self.population.len();
        let mut rng = weak_rng();
        let mut position = worm
            .position
            .iter()
            .enumerate()
            .map(|(j, value)| {
                let sum_j: f64 = self.population.iter().map(|p| p.position[j as usize]).sum();
                let average_j = sum_j / population_size as f64;
                let r = rng.next_f64();
                value + average_j * cauchy(r, 1.0)
            })
            .collect();
        limit_position_random(
            &mut position,
            self.config.lower_bound,
            self.config.upper_bound,
        );
        let fitness = self.calculate_fitness(&position);
        Worm { position, fitness }
    }
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let mut worms = Worms::new(&config, &fitness_evaluator);
    worms.population = worms.generate_population(config.population);
    let elites = 2;
    for iteration in 0..config.iterations {
        worms.sort_population();
        let mut new_worms = vec![];
        for (worm_index, worm) in worms.population.iter().enumerate() {
            let offspring1 = worms.reproduction1(&worm);
            let offspring2 = if worm_index >= elites {
                worms.reproduction2()
            } else {
                worms.random_other_worm(worm_index).position
            };
            let mut new_worm = worms.combine_worms(&offspring1, &offspring2, iteration);
            new_worms.push(new_worm);
        }
        let mut rng = weak_rng();
        for worm in &mut new_worms[elites..] {
            let r = rng.next_f64();
            if 0.01 > r {
                *worm = worms.cauchy_mutation(&worm);
            }
        }
        new_worms.sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        worms.population = worms.population[..elites].to_vec();
        worms
            .population
            .append(&mut new_worms[..config.population - elites].to_vec());

        fitness_evaluator
            .sampler
            .population_sample_single(iteration, &worms.population);
        if fitness_evaluator.end_criteria() {
            break;
        }
    }
    fitness_evaluator
        .sampler
        .population_sample_single(config.iterations, &worms.population);
    solutions_to_json(worms.population)
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::utils::{create_evaluator, create_sampler};

    fn create_config() -> Config {
        Config {
            lower_bound: -4.0,
            upper_bound: 4.0,
            dimensions: 2,
            iterations: 50,
            population: 50,
            beta: 1.0,
            similarity: 0.98,
            cooling_factor: 0.9,
        }
    }

    #[test]
    fn sorts_population_by_ascending_fitness() {
        let config = create_config();
        let sampler = create_sampler();
        let evaluator = create_evaluator(&sampler);
        let mut worms = Worms::new(&config, &evaluator);
        let dimensions = config.dimensions;
        let worm1 = worms.create_worm(vec![0.3; dimensions]);
        let worm2 = worms.create_worm(vec![0.2; dimensions]);
        let worm3 = worms.create_worm(vec![0.02; dimensions]);
        let worm4 = worms.create_worm(vec![0.4; dimensions]);
        worms.population = vec![worm1.clone(), worm2.clone(), worm3.clone(), worm4.clone()];

        worms.sort_population();

        assert_eq!(worms.population, vec![worm3, worm2, worm1, worm4]);
    }

    #[test]
    fn reproduction1_generates_offspring() {
        let config = create_config();
        let sampler = create_sampler();
        let evaluator = create_evaluator(&sampler);
        let worms = Worms::new(&config, &evaluator);
        let dimensions = config.dimensions;
        let worm1 = worms.create_worm(vec![0.3; dimensions]);

        let offspring = worms.reproduction1(&worm1);

        assert_eq!(offspring, vec![-0.3 * config.similarity; dimensions]);
    }

    #[test]
    fn combines_worms_initial() {
        let config = create_config();
        let sampler = create_sampler();
        let evaluator = create_evaluator(&sampler);
        let worms = Worms::new(&config, &evaluator);
        let dimensions = config.dimensions;
        let worm1 = vec![1.0; dimensions];
        let worm2 = vec![2.0; dimensions];

        let combined = worms.combine_worms(&worm1, &worm2, 0);

        assert_eq!(combined.position, vec![1.0; dimensions]);
    }

    #[test]
    fn combines_worms_iteration2() {
        let config = create_config();
        let sampler = create_sampler();
        let evaluator = create_evaluator(&sampler);
        let worms = Worms::new(&config, &evaluator);
        let dimensions = config.dimensions;
        let worm1 = vec![1.0; dimensions];
        let worm2 = vec![2.0; dimensions];

        let combined = worms.combine_worms(&worm1, &worm2, 2);

        assert_eq!(combined.position, vec![1.19; dimensions]);
    }

    #[test]
    fn selects_random_other_worm() {
        let mut config = create_config();
        config.population = 3;
        let sampler = create_sampler();
        let evaluator = create_evaluator(&sampler);
        let mut worms = Worms::new(&config, &evaluator);
        worms.population = worms.generate_population(config.population);
        let worm_index = 1;

        let other_worm = worms.random_other_worm(worm_index);

        let initial_worm = worms.population[worm_index].clone();
        assert!(
            other_worm != initial_worm,
            "Other worm {:?} should not equal initial worm {:?}",
            other_worm,
            initial_worm
        );
    }
}
