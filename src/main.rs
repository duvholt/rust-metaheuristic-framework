#[macro_use]
extern crate clap;
extern crate rustoa;
extern crate serde_json;

use clap::{App, Arg, ArgMatches};
use rustoa::algorithms::da;
use rustoa::algorithms::dummy;
use rustoa::algorithms::ewa;
use rustoa::algorithms::mopso;
use rustoa::algorithms::pso;
use rustoa::algorithms::sa;
use rustoa::config::CommonConfig;
use rustoa::fitness_evaluation::{get_multi, get_single, FitnessEvaluator, TestFunctionVar};
use rustoa::solution::{SolutionJSON, Solutions};
use rustoa::statistics::sampler::{Sampler, SamplerMode};
use rustoa::test_functions;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

fn write_solutions(filename: &str, solutions: Vec<SolutionJSON>, test_function: String) {
    println!("Writing solutions to {}", filename);
    let mut file = File::create(filename).unwrap();
    let solutions_struct = Solutions {
        solutions,
        test_function,
    };
    let json_solutions = serde_json::to_string(&solutions_struct).unwrap();
    file.write_all(json_solutions.as_bytes()).unwrap();
}

type AlgorithmSubCommand = fn(&str) -> App<'static, 'static>;
type AlgorithmRun<S> = fn(&CommonConfig, &FitnessEvaluator<S>, &ArgMatches) -> Vec<SolutionJSON>;
enum AlgorithmType {
    Single(AlgorithmRun<f64>),
    Multi(AlgorithmRun<Vec<f64>>),
}

fn main() {
    let mut algorithms: HashMap<&str, (AlgorithmSubCommand, AlgorithmType)> = HashMap::new();
    algorithms.insert(
        "da",
        (da::subcommand, AlgorithmType::Single(da::run_subcommand)),
    );
    algorithms.insert(
        "dummy",
        (
            dummy::subcommand,
            AlgorithmType::Single(dummy::run_subcommand),
        ),
    );
    algorithms.insert(
        "ewa",
        (ewa::subcommand, AlgorithmType::Single(ewa::run_subcommand)),
    );
    algorithms.insert(
        "pso",
        (pso::subcommand, AlgorithmType::Single(pso::run_subcommand)),
    );
    algorithms.insert(
        "sa",
        (sa::subcommand, AlgorithmType::Single(sa::run_subcommand)),
    );
    algorithms.insert(
        "mopso",
        (
            mopso::subcommand,
            AlgorithmType::Multi(mopso::run_subcommand),
        ),
    );

    let mut test_functions_map = HashMap::new();
    // Single-objective
    test_functions_map.insert(
        "rosenbrock",
        TestFunctionVar::Single(test_functions::rosenbrock),
    );
    test_functions_map.insert(
        "zakharov",
        TestFunctionVar::Single(test_functions::zakharov),
    );
    test_functions_map.insert("ackley", TestFunctionVar::Single(test_functions::ackley));
    test_functions_map.insert(
        "himmelblau",
        TestFunctionVar::Single(test_functions::himmelblau),
    );
    test_functions_map.insert("sphere", TestFunctionVar::Single(test_functions::sphere));
    test_functions_map.insert(
        "rastrigin",
        TestFunctionVar::Single(test_functions::rastrigin),
    );
    test_functions_map.insert(
        "hyper-ellipsoid",
        TestFunctionVar::Single(test_functions::axis_parallel_hyper_ellipsoid),
    );
    test_functions_map.insert(
        "moved-hyper-ellipsoid",
        TestFunctionVar::Single(test_functions::moved_axis_parallel_hyper_ellipsoid),
    );
    // Multi-objective
    test_functions_map.insert(
        "schaffer1",
        TestFunctionVar::Multi(test_functions::schaffer1),
    );
    test_functions_map.insert("zdt1", TestFunctionVar::Multi(test_functions::zdt1));
    test_functions_map.insert("zdt2", TestFunctionVar::Multi(test_functions::zdt2));
    test_functions_map.insert("zdt3", TestFunctionVar::Multi(test_functions::zdt3));
    test_functions_map.insert("zdt6", TestFunctionVar::Multi(test_functions::zdt6));
    test_functions_map.insert("dtlz1", TestFunctionVar::Multi(test_functions::dtlz1));

    let subcommands: Vec<_> = algorithms
        .iter()
        .map(|(name, &(subcommand, _))| subcommand(name))
        .collect();
    let test_function_names: Vec<_> = test_functions_map.keys().map(|&k| k).collect();
    let matches = App::new("Simple Simulated Annealing implementation in Rust using Rosenbrock")
        .arg(
            Arg::with_name("test_function")
                .short("f")
                .long("test-function")
                .value_name("test_function")
                .help("Name of test function")
                .required(true)
                .possible_values(&test_function_names)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("iterations")
                .short("i")
                .long("iterations")
                .value_name("iterations")
                .help("Max iterations")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("evaluations")
                .short("e")
                .long("evaluations")
                .value_name("evaluations")
                .help("Number of fitness evaluations algorithm will run for")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("upper_bound")
                .short("u")
                .long("ub")
                .value_name("upper_bound")
                .help("Upper bound solution space")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("lower_bound")
                .short("l")
                .long("lb")
                .value_name("lower_bound")
                .help("Lower bound solution space")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("dimension")
                .short("d")
                .long("dimension")
                .value_name("dimension")
                .help("Solution dimension size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("population")
                .short("p")
                .long("population")
                .value_name("population")
                .help("Population size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Verbose output"),
        )
        .subcommands(subcommands)
        .get_matches();

    let test_function_name = value_t!(matches, "test_function", String).unwrap();
    let test_function = test_functions_map
        .get(test_function_name.as_str())
        .unwrap()
        .clone();

    let upper_bound = value_t!(matches, "upper_bound", f64).unwrap_or(4.0);
    let common = CommonConfig {
        verbose: matches.is_present("verbose"),
        evaluations: value_t!(matches, "evaluations", i64).unwrap_or(300000),
        iterations: value_t!(matches, "iterations", i64).unwrap_or(1000),
        upper_bound,
        lower_bound: value_t!(matches, "lower_bound", f64).unwrap_or(-upper_bound),
        dimension: value_t!(matches, "dimension", usize).unwrap_or(2),
        population: value_t!(matches, "population", usize).unwrap_or(50),
    };

    println!(
        "Max iterations: {}, Max fitness evaluations: {}, Bounds: ({}, {}), Function: {}, Population: {}",
        common.iterations,
        common.evaluations,
        common.upper_bound,
        common.lower_bound,
        test_function_name,
        common.population
    );

    let (algorithm_name, sub_m) = matches.subcommand();
    // Lookup algorithm in hashmap or panic with a message
    let &(_, ref run_subcommand) = algorithms
        .get(algorithm_name)
        .unwrap_or_else(|| panic!("Algorithm was not specified!"));
    // Run algorithm
    let (solutions, evaluations) = match run_subcommand {
        &AlgorithmType::Single(run) => {
            let sampler = Sampler::new(20, common.iterations, SamplerMode::Evolution);
            let fitness_evaluator =
                FitnessEvaluator::new(get_single(test_function), common.evaluations, &sampler);
            (
                run(&common, &fitness_evaluator, sub_m.unwrap()),
                fitness_evaluator.evaluations(),
            )
        }
        &AlgorithmType::Multi(run) => {
            let sampler = Sampler::new(20, common.iterations, SamplerMode::Evolution);
            let fitness_evaluator =
                FitnessEvaluator::new(get_multi(test_function), common.evaluations, &sampler);
            (
                run(&common, &fitness_evaluator, sub_m.unwrap()),
                fitness_evaluator.evaluations(),
            )
        }
    };

    println!("Number of fitness evaluations: {}", evaluations);

    if let Some(solution) = solutions.last() {
        println!("Final solution: ({:?}) {:?}", solution.x, solution.fitness);
    }

    write_solutions("solutions.json", solutions, test_function_name);
}
