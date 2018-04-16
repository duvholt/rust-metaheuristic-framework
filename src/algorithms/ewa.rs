use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use distribution::cauchy;
use fitness_evaluation::FitnessEvaluator;
use position::random_position;
use rand::{thread_rng, Rng};
use selection::roulette_wheel;
use solution::{solutions_to_json, Solution, SolutionJSON};
use std::cmp::Ordering;

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("earth worm optimization algorithm")
        .arg(
            Arg::with_name("beta")
                .long("beta")
                .value_name("beta")
                .help("beta constant")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("similarity")
                .long("similarity")
                .value_name("similarity")
                .help("similarity constant")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<f64>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let beta = value_t!(sub_m, "beta", f64).unwrap_or(1.0);
    let similarity = value_t!(sub_m, "similarity", f64).unwrap_or(0.98);
    println!("Running EWA with beta: {} similarity: {}", beta, similarity);

    let config = Config {
        space: common.upper_bound,
        dimensions: common.dimensions,
        iterations: common.iterations,
        population: common.population,
        beta,
        similarity,
    };
    run(config, &function_evaluator)
}

#[derive(Debug)]
pub struct Config {
    pub space: f64,
    pub dimensions: usize,
    pub iterations: i64,
    pub population: usize,
    pub beta: f64,
    pub similarity: f64,
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
            -self.config.space,
            self.config.space,
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
            .sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
    }

    fn reproduction1(&self, worm: &Worm) -> Worm {
        let minmax = -self.config.space + self.config.space;
        let alpha = self.config.similarity;
        let mut new_position = vec![];
        for j in 0..self.config.dimensions {
            let x_j = minmax - alpha * worm.position[j];
            new_position.push(x_j);
        }
        let fitness = self.calculate_fitness(&new_position);
        Worm {
            position: new_position,
            fitness,
        }
    }

    fn reproduction2(&self) -> Worm {
        let (parent1index, parent1) = roulette_wheel(&self.population);
        let parent2 = self.random_other_worm(parent1index);

        // uniform crossover
        let mut pos1 = vec![];
        let mut pos2 = vec![];
        let mut rng = thread_rng();
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

        let w1 = f2 / (f1 + f2);
        let w2 = f1 / (f1 + f2);

        let mut position = vec![];
        for j in 0..self.config.dimensions {
            position.push(w1 * pos1[j] + w2 * pos2[j]);
        }
        let fitness = self.calculate_fitness(&position);
        Worm { position, fitness }
    }

    fn combine_worms(&self, worm1: &Worm, worm2: &Worm, iteration: i64) -> Worm {
        let beta = 0.9f64.powf(iteration as f64) * self.config.beta;
        let new_position = (0..self.config.dimensions)
            .map(|j| beta * worm1.position[j] + (1.0 - beta) * worm2.position[j])
            .collect();
        let fitness = self.calculate_fitness(&new_position);
        Worm {
            position: new_position,
            fitness,
        }
    }

    fn random_other_worm(&self, worm_index: usize) -> Worm {
        let mut rng = thread_rng();
        let mut other_index = worm_index;
        while other_index == worm_index {
            other_index = rng.gen_range(0, self.config.population);
        }
        self.population[other_index].clone()
    }

    fn cauchy_mutation(&self, worm: &Worm) -> Worm {
        let population_size = self.population.len();
        let mut rng = thread_rng();
        let position = worm.position
            .iter()
            .enumerate()
            .map(|(j, value)| {
                let sum_j: f64 = self.population.iter().map(|p| p.position[j as usize]).sum();
                let average_j = sum_j / population_size as f64;
                let r = rng.next_f64();
                value + average_j * cauchy(r, 1.0)
            })
            .collect();
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
            let offspring1 = worm.clone();
            let offspring2 = if worm_index > elites {
                worms.reproduction2()
            } else {
                // worms.random_other_worm(worm_index)
                worms.reproduction1(&worm)
            };
            let mut new_worm = worms.combine_worms(&offspring1, &offspring2, iteration);
            new_worms.push(new_worm);
        }
        // The following code was introduced when looking at the matlab version of EWA
        // It does not seem to perform any better though
        new_worms
            .sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
        let mut rng = thread_rng();
        for worm in &mut new_worms {
            let r = rng.next_f64();
            if 0.01 > r {
                *worm = worms.cauchy_mutation(&worm);
            }
        }
        new_worms
            .sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
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
            space: 4.0,
            dimensions: 2,
            iterations: 50,
            population: 50,
            beta: 1.0,
            similarity: 0.98,
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

        assert_eq!(
            offspring.position,
            vec![-0.3 * config.similarity; dimensions]
        );
    }

    #[test]
    fn combines_worms_initial() {
        let config = create_config();
        let sampler = create_sampler();
        let evaluator = create_evaluator(&sampler);
        let worms = Worms::new(&config, &evaluator);
        let dimensions = config.dimensions;
        let worm1 = worms.create_worm(vec![1.0; dimensions]);
        let worm2 = worms.create_worm(vec![2.0; dimensions]);

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
        let worm1 = worms.create_worm(vec![1.0; dimensions]);
        let worm2 = worms.create_worm(vec![2.0; dimensions]);

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
