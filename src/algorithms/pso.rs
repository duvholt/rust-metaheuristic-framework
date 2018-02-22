use solution::Solution;
use rand::distributions::{IndependentSample, Range};
use rand;
use std::cmp::Ordering;

type Position = Vec<f64>;
type TestFunction = Fn(&Vec<f64>) -> f64;

pub struct Config {
    space: f64,
    dimension: i32,
    c1: f64,
    c2: f64,
    inertia: f64,
}

#[derive(Clone)]
struct Particle {
    position: Position,
    pbest: Position,
    fitness: f64,
}

struct Swarm<'a> {
    config: &'a Config,
    population: Vec<Particle>,
    leader: Option<Particle>,
    test_function: &'a Fn(&Vec<f64>) -> f64,
}

impl<'a> Swarm<'a> {
    pub fn new(config: &'a Config, test_function: &'a Fn(&Vec<f64>) -> f64) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            test_function,
            leader: None,
        }
    }

    pub fn random_position(&self) -> Position {
        let between = Range::new(-self.config.space, self.config.space);
        let mut rng = rand::thread_rng();
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
                let fitness = self.calculate_fitness(&position);
                Particle {
                    position: position.to_vec(),
                    pbest: position,
                    fitness,
                }
            })
            .collect()
    }

    fn find_leader(population: &Vec<Particle>) -> Option<&Particle> {
        population
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
    }

    pub fn update_leader(&mut self) {
        self.leader = Swarm::find_leader(&self.population).cloned();
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
}

pub fn run(config: Config, test_function: &'static TestFunction) -> Vec<Solution> {
    let swarm = Swarm::new(&config, &test_function);

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
                pbest: vec![0.0, 1.0],
                fitness: 1.0,
            },
            Particle {
                position: vec![0.0, 1.0],
                pbest: vec![0.0, 1.0],
                fitness: 0.001,
            },
            Particle {
                position: vec![0.0, 1.0],
                pbest: vec![0.0, 1.0],
                fitness: 2.0,
            },
        ];

        let leader = Swarm::find_leader(&population).unwrap();

        assert_eq!(leader.fitness, 0.001);
    }
}
