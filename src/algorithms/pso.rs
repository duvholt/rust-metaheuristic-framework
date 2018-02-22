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
}

impl<'a> Swarm<'a> {
    fn new(config: &'a Config, test_function: &'a Fn(&Vec<f64>) -> f64) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            test_function,
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

    fn find_leader(population: &Vec<Particle>) -> Option<&Particle> {
        population
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
    }

    pub fn solutions(&self) -> Vec<Solution> {
        self.population
            .iter()
            .map(|particle| Solution {
                x: particle.position.to_vec(),
                fitness: particle.fitness,
            })
            .collect()
    }

    fn update_positions(&mut self) {
        let mut rng = thread_rng();
        let leader = Swarm::find_leader(&self.population).unwrap().clone();
        self.population = self.population
            .iter()
            .map(|particle| {
                let r1 = rng.next_f64();
                let r2 = rng.next_f64();
                let mut velocity = vec![];
                for i in 0..self.config.dimension as usize {
                    let v = particle.velocity[i];
                    let x = particle.position[i];
                    let x_p = particle.pbest[i];
                    let x_l = leader.position[i];

                    let new_v = self.config.inertia * v + self.config.c1 * r1 * (x_p - x)
                        + self.config.c2 * r2 * (x_l - x);
                    velocity.push(new_v);
                }
                let position: Vec<f64> = particle
                    .position
                    .iter()
                    .zip(velocity.iter())
                    .map(|(x, v)| x + v)
                    .collect();
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
            })
            .collect();
    }
}

pub fn run(config: Config, test_function: &TestFunction) -> Vec<Solution> {
    let mut swarm = Swarm::new(&config, &test_function);
    swarm.population = swarm.generate_population(100);
    let mut i = 0;
    while i < config.iterations {
        swarm.update_positions();
        i += 1;
        if i % (config.iterations / 20) == 0 {
            let leader = Swarm::find_leader(&swarm.population).unwrap();
            println!("Leader({:?}) fitness {}", leader.position, leader.fitness);
        }
    }
    swarm.solutions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions::rosenbrock;

    #[test]
    fn creates_random_solution() {
        let config = Config {
            space: 4.0,
            dimension: 2,
            iterations: 1,
            c1: 0.2,
            c2: 0.3,
            inertia: 0.5,
        };
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
        let config = Config {
            space: 4.0,
            dimension: 2,
            c1: 0.2,
            c2: 0.3,
            inertia: 0.5,
            iterations: 10,
        };
        let swarm = Swarm::new(&config, &rosenbrock);

        let population = swarm.generate_population(10);

        assert_eq!(population.len(), 10);
    }

    #[test]
    fn finds_leader() {
        let population = vec![
            Particle {
                position: vec![0.0, 1.0],
                velocity: vec![0.0, 1.0],
                pbest: vec![0.0, 1.0],
                fitness: 1.0,
            },
            Particle {
                position: vec![0.0, 1.0],
                velocity: vec![0.0, 1.0],
                pbest: vec![0.0, 1.0],
                fitness: 0.001,
            },
            Particle {
                position: vec![0.0, 1.0],
                velocity: vec![0.0, 1.0],
                pbest: vec![0.0, 1.0],
                fitness: 2.0,
            },
        ];

        let leader = Swarm::find_leader(&population).unwrap();

        assert_eq!(leader.fitness, 0.001);
    }
}
