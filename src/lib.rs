#![feature(test)]
extern crate rand;
extern crate test;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate assert_approx_eq;
#[macro_use]
extern crate serde_derive;
use std::fs::File;
use std::io::prelude::*;

use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng};
mod test_functions;

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

struct Neighbourhood {
    space: f64,
    rng: rand::ThreadRng,
}

impl Neighbourhood {
    pub fn new(space: f64) -> Neighbourhood {
        return Neighbourhood {
            space,
            rng: rand::thread_rng()
        }
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
        )
    }
}

#[derive(Serialize)]
struct Solutions {
    solutions: Vec<Solution>,
}

#[derive(Serialize)]
pub struct Solution {
    pub x: f64,
    pub y: f64,
    pub fitness: f64,
}

impl Solution {
    pub fn new(x: f64, y: f64) -> Solution {
        let fitness = test_functions::ackley(x, y);
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

fn random_solution(config: &Config) -> Solution {
    let between = Range::new(-config.space, config.space);
    let mut rng = rand::thread_rng(); 
    Solution::new(
        between.ind_sample(&mut rng),
        between.ind_sample(&mut rng),
    )
}

pub fn run(config: Config) -> Solution {
    let mut t = config.start_t;
    let mut current = random_solution(&config);
    let mut i = 0;
    let mut rng = thread_rng();
    let mut neighbourhood = Neighbourhood::new(config.space);
    let mut best = current.clone();
    let mut solutions = Solutions {
        solutions: vec![],
    };
    while i < config.iterations {
        t *= config.cooldown;
        let new_solution = neighbourhood.find(&current);
        if new_solution.fitness == 0.0 {
            // Absolute best solution found
            return new_solution;
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
    println!("Diff {} {} {}", current.fitness, best.fitness, current.fitness - best.fitness);
    let mut file = File::create("solutions.json").unwrap();
    let json_solutions = serde_json::to_string(&solutions).unwrap();
    file.write_all(json_solutions.as_bytes()).unwrap();
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;


    #[test]
    fn generates_neighbour() {
        let solution = Solution::new(1., 4.);

        let mut neighbourhood = Neighbourhood::new(1.0);
        let neighbour = neighbourhood.find(&solution);

        assert!(neighbour.x < (solution.x + 1.0) && neighbour.x > (solution.x - 1.0));
        assert!(neighbour.y < (solution.y + 1.0) && neighbour.y > (solution.y - 1.0));
    }

    #[bench]
    fn test_run(b: &mut Bencher) {
        b.iter(|| {
            let config = Config::new(1.0, 0.9, 1000, 4.0);
            run(config);
        });
    }
}
