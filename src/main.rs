extern crate ansi_term;
#[macro_use]
extern crate clap;
extern crate rustoa;
extern crate serde_json;

use ansi_term::Color::{Blue, Cyan, Green, Red};
use clap::{App, Arg, ArgGroup, ArgMatches};
use rustoa::algorithms;
use rustoa::algorithms::{AlgorithmInfo, AlgorithmSubCommand, AlgorithmType};
use rustoa::config::CommonConfig;
use rustoa::fitness_evaluation::{get_multi, get_single, FitnessEvaluator, TestFunctionVar};
use rustoa::problems;
use rustoa::problems::multi::dtlz;
use rustoa::problems::multi::zdt;
use rustoa::problems::single::cec2014;
use rustoa::solution::{Objective, SolutionJSON, Solutions};
use rustoa::statistics::sampler::{Sampler, SamplerMode};
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::stdout;
use std::process;

fn write_solutions(
    filename: &str,
    solutions: Vec<SolutionJSON>,
    test_function: String,
    plot_bounds: bool,
    upper_bound: f64,
    lower_bound: f64,
) {
    println!("Writing solutions to {}", filename);
    let mut file = File::create(filename).unwrap();
    let solutions_struct = Solutions {
        solutions,
        test_function,
        plot_bounds,
        upper_bound,
        lower_bound,
    };
    let json_solutions = serde_json::to_string(&solutions_struct).unwrap();
    file.write_all(json_solutions.as_bytes()).unwrap();
}

pub fn read_pareto_front(filename: &str) -> Vec<Vec<f64>> {
    let file = File::open(filename).unwrap();
    serde_json::from_reader(file).unwrap()
}

fn arguments(
    test_functions_map: &HashMap<&str, TestFunctionVar>,
    algorithms: &HashMap<&str, (AlgorithmSubCommand, AlgorithmType)>,
    test_suites: &HashMap<&str, Vec<String>>,
) -> ArgMatches<'static> {
    // Create a subcommand for each algorithm
    let subcommands: Vec<_> = algorithms
        .iter()
        .map(|(name, &(subcommand, _))| subcommand(name))
        .collect();
    // Create possible values for test functions based on test function hashmap
    let test_function_names: Vec<_> = test_functions_map.keys().map(|&k| k).collect();
    let test_suite_names: Vec<_> = test_suites.keys().map(|&k| k).collect();
    App::new(
        "Various Evolutionary algorithm implementations in Rust evaluated using test functions",
    ).arg(
        Arg::with_name("test_function")
            .short("f")
            .long("test-function")
            .value_name("test_function")
            .help("Name of test function")
            .possible_values(&test_function_names)
            .multiple(true)
            .takes_value(true),
    )
        .arg(
            Arg::with_name("test_suite")
                .short("t")
                .long("test-suite")
                .value_name("test_suite")
                .help("Name of test function suite")
                .possible_values(&test_suite_names)
                .multiple(true)
                .takes_value(true),
        )
        .group(
            ArgGroup::with_name("test")
                .args(&["test_function", "test_suite"])
                .required(true),
        )
        .arg(
            Arg::with_name("runs")
                .short("r")
                .long("runs")
                .value_name("INTEGER")
                .help("Number of runs to perform")
                .default_value("1")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("iterations")
                .short("i")
                .long("iterations")
                .value_name("iterations")
                .help("Max iterations")
                .default_value("1000")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("evaluations")
                .short("e")
                .long("evaluations")
                .value_name("evaluations")
                .help("Number of fitness evaluations algorithm will run for")
                .default_value("300000")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("upper_bound")
                .short("u")
                .long("ub")
                .value_name("upper_bound")
                .help("Upper bound solution space")
                .allow_hyphen_values(true)
                .default_value("30.0")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("lower_bound")
                .short("l")
                .long("lb")
                .value_name("lower_bound")
                .help("Lower bound solution space")
                .allow_hyphen_values(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("dimensions")
                .short("d")
                .long("dimensions")
                .value_name("dimensions")
                .help("Solution dimensions size")
                .default_value("2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("population")
                .short("p")
                .long("population")
                .value_name("population")
                .help("Population size")
                .default_value("50")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("sampler_mode")
                .long("sampler-mode")
                .value_name("sampler_mode")
                .help("Sampling mode")
                .possible_values(&["last", "evolution", "best", "fitness"])
                .default_value("last")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("samples")
                .short("s")
                .long("samples")
                .value_name("samples")
                .help("Number of samples")
                .default_value("30")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("plot-bounds")
                .long("plot-bounds")
                .help("Extends the plot to include the upper and lower bounds"),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .multiple(true)
                .help("Verbose output"),
        )
        .arg(
            Arg::with_name("json")
                .short("j")
                .long("json")
                .help("Output json data about run"),
        )
        .arg(
            Arg::with_name("shift")
                .long("sr")
                .help("Shift and rotate functions (requires data files)"),
        )
        .subcommands(subcommands)
        .get_matches()
}

fn run_algorithm(
    algorithm: &AlgorithmType,
    sub_m: &ArgMatches,
    test_function: TestFunctionVar,
    sampler: &mut Sampler,
    common: &CommonConfig,
    number_of_runs: usize,
    algorithm_info: Option<AlgorithmInfo>,
) -> Result<Vec<SolutionJSON>, &'static str> {
    for run in 0..number_of_runs {
        if common.verbose >= 1 {
            println!("Starting run #{}", Blue.paint(run.to_string()));
        }

        // Run algorithm
        let (_, evaluations) = match algorithm {
            &AlgorithmType::Single(run) => {
                let single_test_function = get_single(test_function)?;
                let mut fitness_evaluator =
                    FitnessEvaluator::new(single_test_function, common.evaluations, &sampler);
                if let Some(ref info) = algorithm_info {
                    let algorithm_number = info.number;
                    fitness_evaluator
                        .read_shifted(algorithm_number, common.dimensions)
                        .unwrap();
                    fitness_evaluator
                        .read_rotate(algorithm_number, common.dimensions)
                        .unwrap();
                    fitness_evaluator.add_to_position = info.add;
                    fitness_evaluator.input_scale = info.scale;
                }
                (
                    run(&common, &fitness_evaluator, sub_m),
                    fitness_evaluator.evaluations(),
                )
            }
            &AlgorithmType::Multi(run) => {
                let (multi_test_function, pareto_filename) = get_multi(test_function)?;
                sampler.set_pareto_front(read_pareto_front(&format!(
                    "optimal_solutions/{}.json",
                    pareto_filename
                )));
                let fitness_evaluator =
                    FitnessEvaluator::new(multi_test_function, common.evaluations, &sampler);
                (
                    run(&common, &fitness_evaluator, sub_m),
                    fitness_evaluator.evaluations(),
                )
            }
        };
        if common.verbose >= 1 {
            sampler.print_run_statistics(stdout());
            println!(
                "Number of fitness evaluations: {}",
                Green.paint(evaluations.to_string())
            );
        }

        sampler.save_run();

        // Keep last run for plotting
        if run + 1 != number_of_runs {
            sampler.end_run();
        }
    }

    sampler.print_statistics(stdout());

    Ok(sampler.solutions())
}

fn start_algorithm() -> Result<(), &'static str> {
    let mut algorithms: HashMap<&str, (AlgorithmSubCommand, AlgorithmType)> = HashMap::new();
    algorithms::add_algorithms(&mut algorithms);

    let mut test_functions_map = HashMap::new();
    // Single-objective
    problems::single::misc::add_test_functions(&mut test_functions_map);
    cec2014::add_test_functions(&mut test_functions_map);
    // Multi-objective
    problems::multi::misc::add_test_functions(&mut test_functions_map);
    dtlz::add_test_functions(&mut test_functions_map);
    zdt::add_test_functions(&mut test_functions_map);

    let mut test_suites = HashMap::new();
    zdt::add_test_suite(&mut test_suites);
    dtlz::add_test_suite(&mut test_suites);
    cec2014::add_test_suite(&mut test_suites);
    problems::single::add_test_suite(&mut test_suites);

    let matches = arguments(&test_functions_map, &algorithms, &test_suites);

    // Test function
    let test_function_names = if matches.is_present("test_function") {
        values_t_or_exit!(matches, "test_function", String)
    } else {
        let test_suite = value_t_or_exit!(matches, "test_suite", String);
        test_suites
            .get(test_suite.as_str())
            .ok_or("Invalid test suite")?
            .clone()
    };
    let number_of_runs = value_t_or_exit!(matches, "runs", usize);

    // Common config for all algorithms
    let upper_bound = value_t_or_exit!(matches, "upper_bound", f64);
    let common = CommonConfig {
        verbose: matches.occurrences_of("verbose"),
        evaluations: value_t_or_exit!(matches, "evaluations", i64),
        iterations: value_t_or_exit!(matches, "iterations", i64),
        upper_bound,
        lower_bound: value_t!(matches, "lower_bound", f64).unwrap_or(-upper_bound),
        dimensions: value_t_or_exit!(matches, "dimensions", usize),
        population: value_t_or_exit!(matches, "population", usize),
    };

    // Sampler settings
    let sampler_mode_name = value_t_or_exit!(matches, "sampler_mode", String);
    let mut samples = value_t_or_exit!(matches, "samples", i64);
    if samples > common.iterations {
        samples = common.iterations;
    }

    let (algorithm_name, sub_m) = matches.subcommand();
    // Lookup algorithm in hashmap or panic with a message
    let &(_, ref run_subcommand) = algorithms
        .get(algorithm_name)
        .ok_or("Algorithm was not specified!")?;

    // Select sampler mode
    let sampler_mode = match sampler_mode_name.as_ref() {
        "last" => SamplerMode::LastGeneration,
        "evolution" => SamplerMode::Evolution,
        "best" => SamplerMode::EvolutionBest,
        "fitness" => SamplerMode::FitnessSearch,
        _ => SamplerMode::LastGeneration,
    };
    let sampler_objective = match run_subcommand {
        &AlgorithmType::Single(_) => Objective::Single,
        &AlgorithmType::Multi(_) => {
            if let SamplerMode::EvolutionBest = sampler_mode {
                return Err("Sampler mode best does not work with multi-objective algorithms");
            }
            Objective::Multi
        }
    };
    let plot_bounds = matches.is_present("plot-bounds");

    println!(
        "Running algorithm {} with bounds ({}, {}) and {} dimensions",
        Green.paint(algorithm_name.to_owned()),
        Green.paint(common.lower_bound.to_string()),
        Green.paint(common.upper_bound.to_string()),
        Green.paint(common.dimensions.to_string()),
    );

    println!(
        "Using a population of {} and {} iterations for a maximum of {} fitness evalutions",
        Blue.paint(common.population.to_string()),
        Blue.paint(common.iterations.to_string()),
        Blue.paint(common.evaluations.to_string()),
    );

    let mut test_function_samples = vec![];

    for test_function_name in test_function_names {
        println!(
            "Running algorithm on {} {} times",
            Cyan.underline().paint(test_function_name.to_string()),
            Green.paint(number_of_runs.to_string()),
        );
        let test_function = test_functions_map
            .get(test_function_name.as_str())
            .unwrap()
            .clone();
        let mut sampler = Sampler::new(
            samples,
            common.iterations,
            sampler_mode.clone(),
            sampler_objective.clone(),
        );
        let algorithm_info = if matches.is_present("shift") {
            cec2014::algorithm_shift_info(test_function_name.as_str())
        } else {
            None
        };

        let solutions = run_algorithm(
            &run_subcommand,
            sub_m.unwrap(),
            test_function,
            &mut sampler,
            &common,
            number_of_runs,
            algorithm_info,
        )?;

        let mut json_sample = sampler.to_json();
        json_sample.test_function = test_function_name.to_string();
        test_function_samples.push(json_sample);

        write_solutions(
            "solutions.json",
            solutions,
            test_function_name,
            plot_bounds,
            common.upper_bound,
            common.lower_bound,
        );
    }

    if matches.is_present("json") {
        println!("{}", serde_json::to_string(&test_function_samples).unwrap());
    }

    Ok(())
}

fn main() {
    if let Err(e) = start_algorithm() {
        eprintln!("{}: {}", Red.paint("Error"), e);
        process::exit(1);
    }
}
