use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use fitness_evaluation::FitnessEvaluator;
use rand;
use rand::distributions::{IndependentSample, Range};
use rand::{weak_rng, Rng};
use solution::{Solution, SolutionJSON};

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("simulated annealing")
        .arg(
            Arg::with_name("start_t")
                .short("t")
                .long("temperature")
                .value_name("start_t")
                .help("Start temperature")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("cooldown")
                .short("-c")
                .long("cooldown")
                .value_name("cooldown")
                .help("Cooldown rate")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<f64>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let start_t = value_t!(sub_m, "start_t", f64).unwrap_or(1.0);
    let cooldown = value_t!(sub_m, "cooldown", f64).unwrap_or(0.9);

    println!(
        "Running SA with start T: {}, cooldown: {}",
        start_t, cooldown
    );
    let config = Config::new(
        start_t,
        cooldown,
        common.iterations,
        common.upper_bound,
        common.dimensions,
    );

    run(config, &function_evaluator)
}

pub struct Config {
    pub start_t: f64,
    pub cooldown: f64,
    pub iterations: i64,
    pub space: f64,
    pub dimensions: usize,
}

impl Config {
    pub fn new(
        start_t: f64,
        cooldown: f64,
        iterations: i64,
        space: f64,
        dimensions: usize,
    ) -> Config {
        return Config {
            start_t,
            cooldown,
            iterations,
            space,
            dimensions,
        };
    }
}

#[derive(Clone)]
struct SASolution {
    x: Vec<f64>,
    fitness: f64,
}

impl Solution<f64> for SASolution {
    fn position(&self) -> &Vec<f64> {
        &self.x
    }
    fn fitness(&self) -> &f64 {
        &self.fitness
    }
}

struct Neighbourhood<'a> {
    dimonension: usize,
    space: f64,
    rng: rand::XorShiftRng,
    fitness_evaluator: &'a FitnessEvaluator<'a, f64>,
}

impl<'a> Neighbourhood<'a> {
    fn new(
        dimonension: usize,
        space: f64,
        fitness_evaluator: &'a FitnessEvaluator<f64>,
    ) -> Neighbourhood<'a> {
        return Neighbourhood {
            dimonension,
            space,
            rng: rand::weak_rng(),
            fitness_evaluator,
        };
    }

    fn random_solution(&self) -> SASolution {
        let between = Range::new(-self.space, self.space);
        let mut rng = rand::weak_rng();
        let x = (0..self.dimonension)
            .map(|_| between.ind_sample(&mut rng))
            .collect();
        let fitness = self.calculate_fitness(&x);
        SASolution { x, fitness }
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        self.fitness_evaluator.calculate_fitness(x)
    }

    fn single_dimensions_neighbour(&mut self, x: &f64) -> f64 {
        let neighbour_space = 0.01;
        let between = Range::new(
            x - self.space * neighbour_space,
            x + self.space * neighbour_space,
        );
        between.ind_sample(&mut self.rng)
    }

    fn find(&mut self, solution: &SASolution) -> SASolution {
        let x = solution
            .x
            .iter()
            .map(|x| self.single_dimensions_neighbour(x))
            .collect();
        let fitness = self.calculate_fitness(&x);
        SASolution { x, fitness }
    }
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let mut t = config.start_t;
    let mut neighbourhood = Neighbourhood::new(config.dimensions, config.space, &fitness_evaluator);
    let mut current = neighbourhood.random_solution();
    let mut i = 0;
    let mut rng = weak_rng();
    let mut best = current.clone();
    let mut solutions = vec![];
    while i < config.iterations {
        t *= config.cooldown;
        let new_solution = neighbourhood.find(&current);
        if new_solution.fitness == 0.0 {
            // Absolute best solution found
            best = new_solution;
            break;
        }
        let delta = current.fitness - new_solution.fitness;
        if delta > 0.0 {
            // Exploit
            current = new_solution;
            best = current.clone();
        } else {
            let a = (delta / t).exp();
            let r = rng.next_f64();
            if a > r {
                // Explore
                current = new_solution;
                if current.fitness < best.fitness {
                    best = current.clone();
                }
            }
        }
        fitness_evaluator
            .sampler
            .population_sample_single(i, &[current.clone()]);
        if fitness_evaluator.end_criteria() {
            break;
        }
        i += 1;
    }
    fitness_evaluator
        .sampler
        .population_sample_single(config.iterations, &[best.clone()]);
    solutions.push(best);
    solutions
        .iter()
        .map(|ref s| SolutionJSON {
            x: s.x.to_vec(),
            fitness: vec![s.fitness],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use testing::utils::{create_evaluator, create_sampler};

    #[test]
    fn generates_neighbour() {
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        let mut neighbourhood = Neighbourhood::new(2, 1.0, &fitness_evaluator);
        let solution = neighbourhood.random_solution();
        let neighbour = neighbourhood.find(&solution);

        let neighbour_x = neighbour.x[0];
        let neighbour_y = neighbour.x[1];
        let solution_x = neighbour.x[0];
        let solution_y = neighbour.x[1];
        assert!(neighbour_x < (solution_x + 1.0) && neighbour_x > (solution_x - 1.0));
        assert!(neighbour_y < (solution_y + 1.0) && neighbour_y > (solution_y - 1.0));
    }

    #[ignore]
    #[bench]
    fn test_run(b: &mut Bencher) {
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        b.iter(|| {
            let config = Config::new(1.0, 0.9, 1000, 4.0, 2);
            run(config, &fitness_evaluator);
        });
    }
}
