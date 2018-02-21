use solution::{Solution, Solutions};

pub struct Config {
    example: f64,
}

impl Config {
    pub fn new(example: f64) -> Config {
        Config { example }
    }
}

pub fn run(
    config: &Config,
    test_function: &Fn(f64, f64) -> f64,
    test_function_name: String,
) -> Solutions {
    println!("Running dummy solver. Example: {}", config.example);
    Solutions {
        solutions: vec![Solution::new(0.0, 0.0, (test_function)(0.0, 0.0))],
        test_function: test_function_name,
    }
}
