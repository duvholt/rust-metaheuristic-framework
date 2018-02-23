use solution::Solution;
use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng};
use std::cmp::Ordering;

type Position = Vec<f64>;
type Velocity = Position;
type TestFunction = Fn(&Vec<f64>) -> f64;

pub struct Config {
    pub space: f64,
    pub dimension: i32,
    pub iterations: i64,
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

struct Swarm<'a> {
    config: &'a Config,
    population: Vec<Particle>,
    test_function: &'a Fn(&Vec<f64>) -> f64,
    leader: Option<Particle>,
}

impl<'a> Swarm<'a> {
    fn new(config: &'a Config, test_function: &'a Fn(&Vec<f64>) -> f64) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            test_function,
            leader: None,
        }
    }

    fn random_position(&self) -> Position {
        let between = Range::new(-self.config.space, self.config.space);
        let mut rng = thread_rng();
        (0..self.config.dimension)
            .map(|_| between.ind_sample(&mut rng))
            .collect()
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        (self.test_function)(x)
    }

    fn generate_population(&self, size: i32) -> Vec<Particle> {
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

    pub fn solutions(&self) -> Vec<Solution> {
        let mut solutions: Vec<Solution> = self.population
            .iter()
            .map(|particle| Solution {
                x: particle.position.to_vec(),
                fitness: particle.fitness,
            })
            .collect();
        let leader = self.get_leader();
        solutions.push(Solution {
            x: leader.position,
            fitness: leader.fitness,
        });
        solutions
            .sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
        solutions
    }

    fn particle_move(&self, particle: &Particle, leader: &Particle) -> Particle {
        let mut rng = thread_rng();
        let r1 = rng.next_f64();
        let r2 = rng.next_f64();
        let mut velocity = vec![];
        let mut position = vec![];
        for i in 0..self.config.dimension as usize {
            let v = particle.velocity[i];
            let x = particle.position[i];
            let x_p = particle.pbest[i];
            let x_l = leader.position[i];

            let new_v = self.config.inertia * v + self.config.c1 * r1 * (x_p - x)
                + self.config.c2 * r2 * (x_l - x);
            velocity.push(new_v);
            position.push(new_v + x);
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

    fn update_positions(&mut self) {
        let leader = self.get_leader();
        self.population = self.population
            .iter()
            .map(|particle| self.particle_move(particle, &leader))
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

pub fn run(config: Config, test_function: &TestFunction) -> Vec<Solution> {
    let mut swarm = Swarm::new(&config, &test_function);
    swarm.population = swarm.generate_population(100);
    let mut i = 0;
    while i < config.iterations {
        swarm.update_leader();
        swarm.update_positions();
        i += 1;
        if i % (config.iterations / 20) == 0 {
            let leader = swarm.get_leader();
            println!("Leader({:?}) fitness {}", leader.position, leader.fitness);
        }
    }
    swarm.solutions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions::rosenbrock;
    use test::Bencher;

    fn create_config() -> Config {
        Config {
            space: 4.0,
            dimension: 2,
            iterations: 20,
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
    fn creates_random_solution() {
        let config = create_config();
        let swarm = Swarm::new(&config, &rosenbrock);

        let position = swarm.random_position();

        assert_eq!(position.len(), config.dimension as usize);
        for coordinate in position {
            println!("{}", coordinate);
            assert!(
                coordinate >= -config.space,
                "Coordinate({}) is outside the allowed solution space({}!",
                coordinate,
                -config.space
            );
            assert!(
                coordinate <= config.space,
                "Coordinate({}) is outside the allowed solution space({}!",
                coordinate,
                config.space
            );
        }
    }

    #[test]
    fn generates_population() {
        let config = create_config();
        let swarm = Swarm::new(&config, &rosenbrock);

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
        let mut swarm = Swarm {
            population: vec![
                create_particle_with_fitness(1.0),
                create_particle_with_fitness(0.001),
            ],
            leader: Some(create_particle_with_fitness(0.02)),
            config: &config,
            test_function: &rosenbrock,
        };

        swarm.update_leader();

        assert_eq!(swarm.leader.unwrap().fitness, 0.001);
    }

    #[test]
    fn does_nothing_if_old_leader_better() {
        let config = create_config();
        let mut swarm = Swarm {
            population: vec![
                create_particle_with_fitness(1.0),
                create_particle_with_fitness(0.1),
            ],
            leader: Some(create_particle_with_fitness(0.02)),
            config: &config,
            test_function: &rosenbrock,
        };

        swarm.update_leader();

        assert_eq!(swarm.leader.unwrap().fitness, 0.02);
    }

    #[bench]
    fn bench_move(b: &mut Bencher) {
        let config = create_config();
        let swarm = Swarm {
            population: vec![
                create_particle_with_fitness(1.0),
                create_particle_with_fitness(0.1),
            ],
            leader: Some(create_particle_with_fitness(0.02)),
            config: &config,
            test_function: &rosenbrock,
        };
        let particle = create_particle_with_fitness(2.0);
        let leader = swarm.get_leader();
        b.iter(|| {
            swarm.particle_move(&particle, &leader);
        });
    }

    #[bench]
    fn bench_pso(b: &mut Bencher) {
        b.iter(|| {
            let config = create_config();
            run(config, &rosenbrock);
        });
    }
}
