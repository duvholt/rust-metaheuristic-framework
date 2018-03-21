use solution::SolutionJSON;
use rand;
use rand::{thread_rng, Rng};
use rand::distributions::{IndependentSample, Range};

pub struct Config {
    pub start_t: f64,
    pub cooldown: f64,
    pub iterations: i64,
    pub space: f64,
    pub dimension: usize,
}

impl Config {
    pub fn new(
        start_t: f64,
        cooldown: f64,
        iterations: i64,
        space: f64,
        dimension: usize,
    ) -> Config {
        return Config {
            start_t,
            cooldown,
            iterations,
            space,
            dimension,
        };
    }
}

#[derive(Clone)]
struct SASolution {
    x: Vec<f64>,
    fitness: f64,
}

struct Neighbourhood<'a> {
    dimonension: usize,
    space: f64,
    rng: rand::ThreadRng,
    test_function: &'a Fn(&Vec<f64>) -> f64,
}

impl<'a> Neighbourhood<'a> {
    fn new(
        dimonension: usize,
        space: f64,
        test_function: &'a Fn(&Vec<f64>) -> f64,
    ) -> Neighbourhood {
        return Neighbourhood {
            dimonension,
            space,
            rng: rand::thread_rng(),
            test_function,
        };
    }

    fn random_solution(&self) -> SASolution {
        let between = Range::new(-self.space, self.space);
        let mut rng = rand::thread_rng();
        let x = (0..self.dimonension)
            .map(|_| between.ind_sample(&mut rng))
            .collect();
        let fitness = self.calculate_fitness(&x);
        SASolution { x, fitness }
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        (self.test_function)(x)
    }

    fn single_dimension_neighbour(&mut self, x: &f64) -> f64 {
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
            .map(|x| self.single_dimension_neighbour(x))
            .collect();
        let fitness = self.calculate_fitness(&x);
        SASolution { x, fitness }
    }
}

pub fn run(config: Config, test_function: &Fn(&Vec<f64>) -> f64) -> Vec<SolutionJSON> {
    let mut t = config.start_t;
    let mut neighbourhood = Neighbourhood::new(config.dimension, config.space, &test_function);
    let mut current = neighbourhood.random_solution();
    let mut i = 0;
    let mut rng = thread_rng();
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
        if i % (config.iterations / 20) == 0 {
            println!("Iterations {} f: {} t: {:0.6}", i, current.fitness, t);
            solutions.push(current.clone());
        }
        i += 1;
    }
    println!(
        "Diff {} {} {}",
        current.fitness,
        best.fitness,
        current.fitness - best.fitness
    );
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
    use test_functions;

    #[test]
    fn generates_neighbour() {
        let test_function = test_functions::rosenbrock;
        let mut neighbourhood = Neighbourhood::new(2, 1.0, &test_function);
        let solution = neighbourhood.random_solution();
        let neighbour = neighbourhood.find(&solution);

        let neighbour_x = neighbour.x[0];
        let neighbour_y = neighbour.x[1];
        let solution_x = neighbour.x[0];
        let solution_y = neighbour.x[1];
        assert!(neighbour_x < (solution_x + 1.0) && neighbour_x > (solution_x - 1.0));
        assert!(neighbour_y < (solution_y + 1.0) && neighbour_y > (solution_y - 1.0));
    }

    #[bench]
    fn test_run(b: &mut Bencher) {
        b.iter(|| {
            let config = Config::new(1.0, 0.9, 1000, 4.0, 2);
            let test_function = test_functions::rosenbrock;
            run(config, &test_function);
        });
    }
}
