use solution::SolutionJSON;

pub struct Config {
    example: f64,
}

impl Config {
    pub fn new(example: f64) -> Config {
        Config { example }
    }
}

pub fn run(config: Config, test_function: &Fn(&Vec<f64>) -> f64) -> Vec<SolutionJSON> {
    println!("Running dummy solver. Example: {}", config.example);
    vec![
        SolutionJSON::new(vec![0.0, 0.0], (test_function)(&vec![0.0, 0.0])),
    ]
}
