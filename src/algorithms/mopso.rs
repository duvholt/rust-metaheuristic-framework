use solution::{solutions_to_json, MultiSolution, SolutionJSON};
use position::random_position;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use domination::{dominates, find_non_dominated};

type Position = Vec<f64>;
type Velocity = Position;
type MultiTestFunction = Fn(&Vec<f64>) -> Vec<f64>;

pub struct Config {
    pub space: f64,
    pub dimension: usize,
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
    fitness: Vec<f64>,
    velocity: Velocity,
}

impl MultiSolution for Particle {
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
    test_function: &'a MultiTestFunction,
    leaders: Vec<Particle>,
}

impl<'a> Swarm<'a> {
    fn new(config: &'a Config, test_function: &'a MultiTestFunction) -> Swarm<'a> {
        Swarm {
            config,
            population: vec![],
            test_function,
            leaders: vec![],
        }
    }

    fn random_position(&self) -> Position {
        random_position(-self.config.space, self.config.space, self.config.dimension)
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> Vec<f64> {
        (self.test_function)(x)
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

    fn find_leader(population: &Vec<Particle>) -> Option<Particle> {
        population
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
            .cloned()
    }

    // pub fn solutions(&self) -> Vec<SolutionJSON> {
    //     let solutions = self.leaders.to_vec();
    //     solutions_to_json(solutions)
    // }

    fn particle_move(&self, particle: &Particle, leader: &Particle) -> Particle {
        let mut rng = thread_rng();
        let r1 = rng.next_f64();
        let r2 = rng.next_f64();
        let mut velocity = vec![];
        let mut position = vec![];
        for i in 0..self.config.dimension {
            let v = particle.velocity[i];
            let x = particle.position[i];
            let x_p = particle.pbest[i];
            let x_l = leader.position[i];

            let mut new_v = self.config.inertia * v + self.config.c1 * r1 * (x_p - x)
                + self.config.c2 * r2 * (x_l - x);
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
            velocity.push(new_v);
            position.push(new_x);
        }
        let fitness = self.calculate_fitness(&position);
        let pbest = if dominates(&fitness, &particle.fitness) {
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
        // TODO: Properly select
        let leader = &self.leaders[0];
        self.population = self.population
            .iter()
            .map(|particle| self.particle_move(particle, &leader))
            .collect();
    }

    fn find_leaders(&self) -> Vec<Particle> {
        let non_dominated = find_non_dominated(&self.population);
        non_dominated.iter().map(|p_i| self.population[*p_i].clone()).collect()
    }
}

pub fn run(config: Config, test_function: &'static MultiTestFunction) { // -> Vec<SolutionJSON> {
    let mut swarm = Swarm::new(&config, &test_function);
    swarm.population = swarm.generate_population(config.population);
    swarm.leaders = swarm.find_leaders();
    let mut i = 0;
    while i < config.iterations {
        swarm.update_positions();
        i += 1;
    }
    //swarm.solutions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions::multi_dummy;
    use test::Bencher;

    fn create_config() -> Config {
        Config {
            space: 4.0,
            dimension: 2,
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
            fitness: vec![fitness, fitness],
        }
    }

    #[test]
    fn generates_population() {
        let config = create_config();
        let swarm = Swarm::new(&config, &multi_dummy);

        let population = swarm.generate_population(10);

        assert_eq!(population.len(), 10);
    }

    #[bench]
    fn bench_move(b: &mut Bencher) {
        let config = create_config();
        let swarm = Swarm {
            population: vec![
                create_particle_with_fitness(1.0),
                create_particle_with_fitness(0.1),
            ],
            leaders: vec![create_particle_with_fitness(0.1)],
            config: &config,
            test_function: &multi_dummy,
        };
        let particle = create_particle_with_fitness(2.0);
        // TODO: Properly select
        let leader = &swarm.leaders[0];
        b.iter(|| {
            swarm.particle_move(&particle, &leader);
        });
    }

    #[bench]
    fn bench_pso(b: &mut Bencher) {
        b.iter(|| {
            let config = create_config();
            run(config, &multi_dummy);
        });
    }
}
