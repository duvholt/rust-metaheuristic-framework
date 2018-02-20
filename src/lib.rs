#![feature(test)]
extern crate rand;
extern crate test;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng};

pub mod test_functions;

pub struct Config {
    pub start_t: f64,
    pub cooldown: f64,
    pub iterations: i64,
    pub space: f64,
}

impl Config {
    pub fn new(start_t: f64, cooldown: f64, iterations: i64, space: f64) -> Config {
        return Config {
            start_t, cooldown, iterations, space
        }
    }
}

struct Neighbourhood<'a>
{
    space: f64,
    rng: rand::ThreadRng,
    test_function: &'a Fn(f64, f64) -> f64,
}

impl<'a> Neighbourhood<'a>
{
    pub fn new<'b>(space: f64, test_function: &'b Fn(f64, f64) -> f64) -> Neighbourhood
    {
        return Neighbourhood {
            space,
            rng: rand::thread_rng(),
            test_function
        }
    }

    fn random_solution(&self) -> Solution {
        let between = Range::new(-self.space, self.space);
        let mut rng = rand::thread_rng();
        let x = between.ind_sample(&mut rng);
        let y = between.ind_sample(&mut rng);
        Solution::new(
            x, y,
            self.calculate_fitness(x, y)
        )
    }

    fn calculate_fitness(&self, x: f64, y: f64) -> f64 {
        (self.test_function)(x, y)
    }

    fn single_dimension_neighbour(&mut self, x: f64) -> f64 {
        let neighbour_space = 0.01;
        let between = Range::new(x - self.space * neighbour_space, x + self.space * neighbour_space);
        between.ind_sample(&mut self.rng)
    }

    fn find(&mut self, solution: &Solution) -> Solution {
        let x = self.single_dimension_neighbour(solution.x);
        let y = self.single_dimension_neighbour(solution.y);
        Solution::new(
            x,
            y,
            self.calculate_fitness(x, y)
        )
    }
}

#[derive(Serialize)]
pub struct Solutions {
    pub solutions: Vec<Solution>,
}

#[derive(Serialize)]
pub struct Solution {
    pub x: f64,
    pub y: f64,
    pub fitness: f64,
}

impl Solution {
    pub fn new(x: f64, y: f64, fitness: f64) -> Solution {
        Solution {
            x,
            y,
            fitness,
        }
    }
}

impl Copy for Solution { }

impl Clone for Solution {
    fn clone(&self) -> Solution {
        *self
    }
}

pub fn run(config: Config, test_function: &Fn(f64, f64) -> f64) -> Solutions {
    let mut t = config.start_t;
    let mut neighbourhood = Neighbourhood::new(config.space, &test_function);
    let mut current = neighbourhood.random_solution();
    let mut i = 0;
    let mut rng = thread_rng();
    let mut best = current.clone();
    let mut solutions = Solutions {
        solutions: vec![],
    };
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
            solutions.solutions.push(current.clone());
        }
        i += 1;
    }
    solutions.solutions.push(best);
    println!("Diff {} {} {}", current.fitness, best.fitness, current.fitness - best.fitness);
    solutions
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;


    #[test]
    fn generates_neighbour() {

        let test_function = test_functions::rosenbrock;
        let mut neighbourhood = Neighbourhood::new(1.0, &test_function);
        let solution = neighbourhood.random_solution();
        let neighbour = neighbourhood.find(&solution);

        assert!(neighbour.x < (solution.x + 1.0) && neighbour.x > (solution.x - 1.0));
        assert!(neighbour.y < (solution.y + 1.0) && neighbour.y > (solution.y - 1.0));
    }

    #[bench]
    fn test_run(b: &mut Bencher) {
        b.iter(|| {
            let config = Config::new(1.0, 0.9, 1000, 4.0);
            let test_function = test_functions::rosenbrock;
            run(config, &test_function);
        });
    }
}
