use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use fitness_evaluation::FitnessEvaluator;
use operators::position::random_position;
use rand::{weak_rng, Rng};
use solution::{solutions_to_json, Solution, SolutionJSON};
use std::cmp::Ordering;

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
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<f64>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let c1 = value_t!(sub_m, "c1", f64).unwrap_or(2.0);
    let c2 = value_t!(sub_m, "c2", f64).unwrap_or(2.0);
    let inertia = value_t!(sub_m, "inertia", f64).unwrap_or(1.1);
    if common.verbose >= 1 {
        println!(
            "Running PSO with C1: {}, C2: {} inertia: {}",
            c1, c2, inertia
        );
    }

    let config = Config {
        space: common.upper_bound,
        dimensions: common.dimensions,
        iterations: common.iterations,
        population: common.population,
        c1,
        c2,
        inertia,
    };
    run(config, &function_evaluator)
}

type Position = Vec<f64>;
type Velocity = Position;

pub struct Config {
    pub space: f64,
    pub dimensions: usize,
    pub iterations: i64,
    pub population: usize,
    pub c1: f64,
    pub c2: f64,
    pub inertia: f64,
}

#[derive(Clone)]
struct Particle {
    position: Position,
    pbest: Position,
    fitness: f64,
    velocity: Velocity,
}

impl Solution<f64> for Particle {
    fn fitness(&self) -> &f64 {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

struct Swarm<'a> {
    config: &'a Config,
    population: Vec<Particle>,
    fitness_evaluator: &'a FitnessEvaluator<'a, f64>,
    leader: Option<Particle>,
}

impl<'a> Swarm<'a> {
    fn new(config: &'a Config, fitness_evaluator: &'a FitnessEvaluator<f64>) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            fitness_evaluator,
            leader: None,
        }
    }

    fn random_position(&self) -> Position {
        random_position(
            -self.config.space,
            self.config.space,
            self.config.dimensions,
        )
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        self.fitness_evaluator.calculate_fitness(x)
    }

    fn generate_population(&self, size: usize) -> Vec<Particle> {
        (0..size)
            .map(|_| {
                let position = self.random_position();
                let velocity = self.random_position();
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

    fn find_leader(population: &Vec<Particle>) -> Option<Particle> {
        population
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
            .cloned()
    }

    pub fn solutions(&self) -> Vec<SolutionJSON> {
        let mut solutions = self.population.to_vec();
        solutions.push(self.get_leader());
        solutions_to_json(solutions)
    }

    fn particle_move(&self, particle: &Particle, leader: &Particle, iteration: i64) -> Particle {
        let mut rng = weak_rng();
        let mut velocity = vec![];
        let mut position = vec![];
        let inertia = self.config.inertia * 0.99_f64.powi(iteration as i32);
        for i in 0..self.config.dimensions {
            let r1 = rng.next_f64();
            let r2 = rng.next_f64();
            let v = particle.velocity[i];
            let x = particle.position[i];
            let x_p = particle.pbest[i];
            let x_l = leader.position[i];

            let mut new_v =
                inertia * v + self.config.c1 * r1 * (x_p - x) + self.config.c2 * r2 * (x_l - x);
            let mut new_x = new_v + x;
            if new_v + x > self.config.space {
                // Bound hit, move in opposite direction
                new_v *= -1.0;
                new_x = self.config.space;
            } else if (new_v + x) < -self.config.space {
                // Bound hit, move in opposite direction
                new_v *= -1.0;
                new_x = -self.config.space;
            }
            let velocity_limit = 10.0;
            if new_v > velocity_limit {
                new_v = velocity_limit;
            }
            if new_v < -velocity_limit {
                new_v = -velocity_limit;
            }
            velocity.push(new_v);
            position.push(new_x);
        }
        let fitness = self.calculate_fitness(&position);
        let pbest = if fitness < particle.fitness {
            position.clone()
        } else {
            particle.pbest.clone()
        };
        Particle {
            position,
            pbest,
            fitness,
            velocity: velocity,
        }
    }

    fn update_positions(&mut self, iteration: i64) {
        let leader = self.get_leader();
        self.population = self.population
            .iter()
            .map(|particle| self.particle_move(particle, &leader, iteration))
            .collect();
    }

    fn get_leader(&self) -> Particle {
        self.leader.as_ref().unwrap().clone()
    }

    fn update_leader(&mut self) {
        let leader = Swarm::find_leader(&self.population).unwrap();
        self.leader = match self.leader.clone() {
            Some(old_leader) => {
                if old_leader.fitness > leader.fitness {
                    Some(leader)
                } else {
                    Some(old_leader)
                }
            }
            None => Some(leader),
        }
    }
}

pub fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let mut swarm = Swarm::new(&config, &fitness_evaluator);
    swarm.population = swarm.generate_population(config.population);
    let mut i = 0;
    while i < config.iterations {
        swarm.update_leader();
        swarm.update_positions(i);
        fitness_evaluator
            .sampler
            .population_sample_single(i, &swarm.population);
        if fitness_evaluator.end_criteria() {
            break;
        }
        i += 1;
    }
    fitness_evaluator
        .sampler
        .population_sample_single(config.iterations, &swarm.population);
    swarm.solutions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use testing::utils::{create_evaluator, create_sampler};

    fn create_config() -> Config {
        Config {
            space: 4.0,
            dimensions: 2,
            iterations: 20,
            population: 100,
            c1: 2.0,
            c2: 2.0,
            inertia: 1.1,
        }
    }

    fn create_particle_with_fitness(fitness: f64) -> Particle {
        Particle {
            position: vec![0.0, 1.0],
            velocity: vec![0.0, 1.0],
            pbest: vec![0.0, 1.0],
            fitness,
        }
    }

    #[test]
    fn generates_population() {
        let config = create_config();
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        let swarm = Swarm::new(&config, &fitness_evaluator);

        let population = swarm.generate_population(10);

        assert_eq!(population.len(), 10);
    }

    #[test]
    fn finds_leader() {
        let population = vec![
            create_particle_with_fitness(1.0),
            create_particle_with_fitness(0.001),
            create_particle_with_fitness(2.0),
        ];

        let leader = Swarm::find_leader(&population).unwrap();

        assert_eq!(leader.fitness, 0.001);
    }

    #[test]
    fn updates_leader_if_better() {
        let config = create_config();
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        let mut swarm = Swarm {
            population: vec![
                create_particle_with_fitness(1.0),
                create_particle_with_fitness(0.001),
            ],
            leader: Some(create_particle_with_fitness(0.02)),
            config: &config,
            fitness_evaluator: &fitness_evaluator,
        };

        swarm.update_leader();

        assert_eq!(swarm.leader.unwrap().fitness, 0.001);
    }

    #[test]
    fn does_nothing_if_old_leader_better() {
        let config = create_config();
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        let mut swarm = Swarm {
            population: vec![
                create_particle_with_fitness(1.0),
                create_particle_with_fitness(0.1),
            ],
            leader: Some(create_particle_with_fitness(0.02)),
            config: &config,
            fitness_evaluator: &fitness_evaluator,
        };

        swarm.update_leader();

        assert_eq!(swarm.leader.unwrap().fitness, 0.02);
    }

    #[ignore]
    #[bench]
    fn bench_move(b: &mut Bencher) {
        let config = create_config();
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        let swarm = Swarm {
            population: vec![
                create_particle_with_fitness(1.0),
                create_particle_with_fitness(0.1),
            ],
            leader: Some(create_particle_with_fitness(0.02)),
            config: &config,
            fitness_evaluator: &fitness_evaluator,
        };
        let particle = create_particle_with_fitness(2.0);
        let leader = swarm.get_leader();
        b.iter(|| {
            swarm.particle_move(&particle, &leader, 0);
        });
    }

    #[ignore]
    #[bench]
    fn bench_pso(b: &mut Bencher) {
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        b.iter(|| {
            let config = create_config();
            run(config, &fitness_evaluator);
        });
    }
}
