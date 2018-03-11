use solution::{Neighbourhood, Solution};
use rand::{thread_rng, Rng};

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

pub fn run(config: Config, test_function: &Fn(&Vec<f64>) -> f64) -> Vec<Solution> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use test_functions;

    #[bench]
    fn test_run(b: &mut Bencher) {
        b.iter(|| {
            let config = Config::new(1.0, 0.9, 1000, 4.0, 2);
            let test_function = test_functions::rosenbrock;
            run(config, &test_function);
        });
    }
}
