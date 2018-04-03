use archive::Archive;
use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use domination::dominates;
use fitness_evaluation::FitnessEvaluator;
use position::random_position;
use rand::{thread_rng, Rng};
use solution::{multi_solutions_to_json, Solution, SolutionJSON};

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("particle swarm optimization")
        .arg(
            Arg::with_name("c1")
                .long("c1")
                .value_name("c1")
                .help("C1 constant")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("c2")
                .long("c2")
                .value_name("c2")
                .help("C2 constant")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("inertia")
                .long("inertia")
                .value_name("inertia")
                .help("inertia constant")
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
        .arg(
            Arg::with_name("mutation_rate")
                .short("-m")
                .long("mutation_rate")
                .value_name("mutation_rate")
                .help("mutation rate")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<Vec<f64>>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let c1 = value_t!(sub_m, "c1", f64).unwrap_or(1.0);
    let c2 = value_t!(sub_m, "c2", f64).unwrap_or(2.0);
    let inertia = value_t!(sub_m, "inertia", f64).unwrap_or(0.4);
    let archive_size = value_t!(sub_m, "archive_size", usize).unwrap_or(common.population);
    let divisions = value_t!(sub_m, "divisions", usize).unwrap_or(30);
    let mutation_rate = value_t!(sub_m, "mutation_rate", f64).unwrap_or(0.1);
    println!(
        "Running MOPSO with C1: {}, C2: {} inertia: {}",
        c1, c2, inertia
    );

    let config = Config {
        upper_bound: common.upper_bound,
        lower_bound: common.lower_bound,
        dimension: common.dimension,
        iterations: common.iterations,
        population: common.population,
        verbose: common.verbose,
        c1,
        c2,
        inertia,
        archive_size,
        divisions,
        mutation_rate,
    };
    run(config, function_evaluator)
}

type Position = Vec<f64>;
type Velocity = Position;

#[derive(Debug)]
pub struct Config {
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub dimension: usize,
    pub iterations: i64,
    pub population: usize,
    pub c1: f64,
    pub c2: f64,
    pub inertia: f64,
    pub archive_size: usize,
    pub divisions: usize,
    pub mutation_rate: f64,
    pub verbose: bool,
}

#[derive(Clone, Debug)]
struct Particle {
    position: Position,
    pbest: Position,
    fitness: Vec<f64>,
    velocity: Velocity,
}

impl Solution<Vec<f64>> for Particle {
    fn fitness(&self) -> &Vec<f64> {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

struct Swarm<'a> {
    config: &'a Config,
    population: Vec<Particle>,
    fitness_evaluator: &'a FitnessEvaluator<'a, Vec<f64>>,
    archive: Archive<Particle>,
}

impl<'a> Swarm<'a> {
    fn new(config: &'a Config, fitness_evaluator: &'a FitnessEvaluator<Vec<f64>>) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            fitness_evaluator,
            archive: Archive::new(config.archive_size, config.divisions),
        }
    }

    fn random_position(&self) -> Position {
        random_position(
            self.config.lower_bound,
            self.config.upper_bound,
            self.config.dimension,
        )
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> Vec<f64> {
        self.fitness_evaluator.calculate_fitness(x)
    }

    fn generate_population(&self, size: usize) -> Vec<Particle> {
        (0..size)
            .map(|_| {
                let position = self.random_position();
                let velocity = vec![0.0; self.config.dimension];
                let fitness = self.calculate_fitness(&position);
                Particle {
                    position: position.to_vec(),
                    pbest: position,
                    fitness,
                    velocity,
                }
            })
            .collect()
    }

    pub fn solutions(&self) -> Vec<SolutionJSON> {
        let solutions = self.archive.population.to_vec();
        multi_solutions_to_json(solutions)
    }

    fn mutate(&self, position: &Vec<f64>, pm: f64) -> Vec<f64> {
        let diff_position = pm * (self.config.upper_bound - self.config.lower_bound);
        let mut rng = thread_rng();
        let j: usize = rng.gen_range(0, self.config.dimension);

        let mut lb = position[j] - diff_position;
        if lb < self.config.lower_bound {
            lb = self.config.lower_bound;
        }

        let mut ub = position[j] + diff_position;
        if ub > self.config.upper_bound {
            ub = self.config.upper_bound;
        }
        let mut mutated_position = position.to_vec();
        mutated_position[j] = rng.gen_range(lb, ub);
        mutated_position
    }

    fn particle_move(&self, particle: &Particle, leader: &Particle, iteration: i64) -> Particle {
        let mut rng = thread_rng();
        let mut velocity = vec![];
        let mut position = vec![];
        for i in 0..self.config.dimension {
            let r1 = rng.next_f64();
            let r2 = rng.next_f64();
            let v = particle.velocity[i];
            let x = particle.position[i];
            let x_p = particle.pbest[i];
            let x_l = leader.position[i];

            let inertia = self.config.inertia.powf(iteration as f64);
            let c1 = self.config.c1;
            let c2 = self.config.c2;
            let mut new_v = inertia * v + c1 * r1 * (x_p - x) + c2 * r2 * (x_l - x);
            let mut new_x = new_v + x;
            if new_v + x > self.config.upper_bound {
                // Bound hit, move in opposite direction
                new_v *= -1.0;
                new_x = self.config.upper_bound;
            } else if (new_v + x) < self.config.lower_bound {
                // Bound hit, move in opposite direction
                new_v *= -1.0;
                new_x = self.config.lower_bound;
            }
            velocity.push(new_v);
            position.push(new_x);
        }
        let mut fitness = self.calculate_fitness(&position);
        // Mutate particle
        let pm = (1.0 - (iteration as f64 / self.config.iterations as f64))
            .powf(1.0 / self.config.mutation_rate);
        if rng.next_f64() < pm {
            let mutated_position = self.mutate(&position, pm);
            let mutated_fitness = self.calculate_fitness(&mutated_position);
            if dominates(&mutated_fitness, &fitness) {
                if self.config.verbose {
                    println!(
                        "Improved solution with mutation. old: {:?} new: {:?}",
                        fitness, mutated_fitness
                    );
                }
                position = mutated_position;
                fitness = mutated_fitness;
            } else if rng.next_f64() > 0.5 && !dominates(&fitness, &mutated_fitness) {
                if self.config.verbose {
                    println!(
                        "Randomly selected mutation. old: {:?} new: {:?}",
                        fitness, mutated_fitness
                    );
                }
                position = mutated_position;
                fitness = mutated_fitness;
            }
        }
        // Select the dominating particle as pbest
        let pbest = if dominates(&fitness, &particle.fitness) {
            position.clone()
        } else if dominates(&particle.fitness, &fitness) {
            particle.pbest.clone()
        } else {
            // If neither dominates the other select randomly
            let r = rng.next_f64();
            if r > 0.5 {
                position.clone()
            } else {
                particle.pbest.clone()
            }
        };
        Particle {
            position,
            pbest,
            fitness,
            velocity: velocity,
        }
    }

    fn update_positions(&mut self, iteration: i64) {
        let leader = self.archive.select_leader();
        self.population = self.population
            .iter()
            .map(|particle| self.particle_move(particle, &leader, iteration))
            .collect();
    }
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<Vec<f64>>) -> Vec<SolutionJSON> {
    let mut swarm = Swarm::new(&config, fitness_evaluator);
    swarm.population = swarm.generate_population(config.population);
    let mut i = 0;
    while i < config.iterations {
        if config.verbose {
            println!(
                "Iteration {} Archive size {}",
                i,
                swarm.archive.population.len()
            );
        }
        swarm.archive.update(&swarm.population);
        swarm.update_positions(i);
        fitness_evaluator
            .sampler
            .population_sample_multi(i, &swarm.population);
        if fitness_evaluator.end_criteria() {
            break;
        }
        i += 1;
    }
    swarm.solutions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use statistics::sampler::{Sampler, SamplerMode};
    use test::Bencher;
    use test_functions::multi_dummy;

    fn create_config() -> Config {
        Config {
            upper_bound: 4.0,
            lower_bound: -4.0,
            dimension: 2,
            iterations: 20,
            population: 50,
            archive_size: 50,
            divisions: 30,
            c1: 2.0,
            c2: 2.0,
            inertia: 1.1,
            mutation_rate: 0.5,
            verbose: false,
        }
    }

    fn create_sampler() -> Sampler {
        Sampler::new(10, 10, SamplerMode::Evolution)
    }

    fn create_evaluator(sampler: &Sampler) -> FitnessEvaluator<Vec<f64>> {
        FitnessEvaluator::new(multi_dummy, 100, &sampler)
    }

    fn create_particle_with_fitness(fitness: f64) -> Particle {
        Particle {
            position: vec![0.0, 1.0],
            velocity: vec![0.0, 1.0],
            pbest: vec![0.0, 1.0],
            fitness: vec![fitness, fitness],
        }
    }

    #[test]
    fn generates_population() {
        let config = create_config();
        let sampler = create_sampler();
        let function_evaluator = create_evaluator(&sampler);
        let swarm = Swarm::new(&config, &function_evaluator);

        let population = swarm.generate_population(10);

        assert_eq!(population.len(), 10);
    }

    #[ignore]
    #[bench]
    fn bench_move(b: &mut Bencher) {
        let config = create_config();
        let population = vec![
            create_particle_with_fitness(1.0),
            create_particle_with_fitness(0.1),
        ];
        let sampler = create_sampler();
        let function_evaluator = create_evaluator(&sampler);
        let mut swarm = Swarm::new(&config, &function_evaluator);
        swarm.population = population;
        let particle = create_particle_with_fitness(2.0);
        let leader = create_particle_with_fitness(0.01);
        b.iter(|| {
            swarm.particle_move(&particle, &leader, 0);
        });
    }

    #[ignore]
    #[bench]
    fn bench_run(b: &mut Bencher) {
        let sampler = create_sampler();
        let function_evaluator = create_evaluator(&sampler);
        b.iter(|| run(create_config(), &function_evaluator));
    }
}
