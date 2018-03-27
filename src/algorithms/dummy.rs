use solution::SolutionJSON;
use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use test_functions::SingleTestFunctionVar;

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name).about("dummy solver").arg(
        Arg::with_name("example")
            .long("example")
            .value_name("example")
            .help("example argument")
            .takes_value(true),
    )
}

pub fn run_subcommand(
    _common: &CommonConfig,
    test_function: SingleTestFunctionVar,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let example = value_t!(sub_m, "example", f64).unwrap_or(1.0);
    println!("Running dummy solver with example: {}", example);
    let config = Config::new(example);

    run(config, &test_function)
}

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
        SolutionJSON::new(vec![0.0, 0.0], vec![(test_function)(&vec![0.0, 0.0])]),
    ]
}
