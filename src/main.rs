extern crate ansi_term;
#[macro_use]
extern crate clap;
extern crate rustoa;
extern crate serde_json;

use ansi_term::Color::{Blue, Green, Red, Yellow};
use clap::{App, Arg, ArgMatches};
use rustoa::algorithms::amo;
use rustoa::algorithms::da;
use rustoa::algorithms::dummy;
use rustoa::algorithms::ewa;
use rustoa::algorithms::loa;
use rustoa::algorithms::mopso;
use rustoa::algorithms::pso;
use rustoa::algorithms::sa;
use rustoa::config::CommonConfig;
use rustoa::fitness_evaluation::{get_multi, get_single, FitnessEvaluator, TestFunctionVar};
use rustoa::solution::{Objective, SolutionJSON, Solutions};
use rustoa::statistics::sampler::{Sampler, SamplerMode};
use rustoa::test_functions;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::stdout;
use std::process;

type AlgorithmSubCommand = fn(&str) -> App<'static, 'static>;
type AlgorithmRun<S> = fn(&CommonConfig, &FitnessEvaluator<S>, &ArgMatches) -> Vec<SolutionJSON>;
enum AlgorithmType {
    Single(AlgorithmRun<f64>),
    Multi(AlgorithmRun<Vec<f64>>),
}

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
) -> ArgMatches<'static> {
    // Create a subcommand for each algorithm
    let subcommands: Vec<_> = algorithms
        .iter()
        .map(|(name, &(subcommand, _))| subcommand(name))
        .collect();
    // Create possible values for test functions based on test function hashmap
    let test_function_names: Vec<_> = test_functions_map.keys().map(|&k| k).collect();
    App::new(
        "Various Evolutionary algorithm implementations in Rust evaluated using test functions",
    ).arg(
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
                .default_value("30.0")
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
                .help("Verbose output"),
        )
        .subcommands(subcommands)
        .get_matches()
}

fn start_algorithm() -> Result<(), &'static str> {
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
        "loa",
        (loa::subcommand, AlgorithmType::Single(loa::run_subcommand)),
    );
    algorithms.insert(
        "mopso",
        (
            mopso::subcommand,
            AlgorithmType::Multi(mopso::run_subcommand),
        ),
    );
    algorithms.insert(
        "amo",
        (amo::subcommand, AlgorithmType::Single(amo::run_subcommand)),
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
        TestFunctionVar::Multi(test_functions::schaffer1, "schaffer1-2d"),
    );
    test_functions_map.insert(
        "zdt1",
        TestFunctionVar::Multi(test_functions::zdt1, "zdt1-2d"),
    );
    test_functions_map.insert(
        "zdt2",
        TestFunctionVar::Multi(test_functions::zdt2, "zdt2-2d"),
    );
    test_functions_map.insert(
        "zdt3",
        TestFunctionVar::Multi(test_functions::zdt3, "zdt3-2d"),
    );
    test_functions_map.insert(
        "zdt6",
        TestFunctionVar::Multi(test_functions::zdt6, "zdt6-2d"),
    );
    test_functions_map.insert(
        "dtlz1",
        TestFunctionVar::Multi(test_functions::dtlz1, "dtlz1-3d"),
    );
    test_functions_map.insert(
        "dtlz2",
        TestFunctionVar::Multi(test_functions::dtlz2, "dtlz2-3d"),
    );
 
    let matches = arguments(&test_functions_map, &algorithms);

    // Test function
    let test_function_name = value_t_or_exit!(matches, "test_function", String);
    let test_function = test_functions_map
        .get(test_function_name.as_str())
        .unwrap()
        .clone();

    // Common config for all algorithms
    let upper_bound = value_t_or_exit!(matches, "upper_bound", f64);
    let common = CommonConfig {
        verbose: matches.is_present("verbose"),
        evaluations: value_t_or_exit!(matches, "evaluations", i64),
        iterations: value_t_or_exit!(matches, "iterations", i64),
        upper_bound,
        lower_bound: value_t!(matches, "lower_bound", f64).unwrap_or(-upper_bound),
        dimensions: value_t_or_exit!(matches, "dimensions", usize),
        population: value_t_or_exit!(matches, "population", usize),
    };

    // Sampler settings
    let sampler_mode_name = value_t_or_exit!(matches, "sampler_mode", String);
    let samples = value_t_or_exit!(matches, "samples", i64);

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
    let mut sampler = Sampler::new(samples, common.iterations, sampler_mode, sampler_objective);

    println!(
        "Running algorithm {} on test function {} with bounds ({}, {}) and {} dimensions",
        Green.paint(algorithm_name.to_owned()),
        Green.paint(test_function_name.to_owned()),
        Green.paint(common.upper_bound.to_string()),
        Green.paint(common.lower_bound.to_string()),
        Green.paint(common.dimensions.to_string()),
    );

    println!(
        "Using a population of {} and {} iterations for a maximum of {} fitness evalutions",
        Blue.paint(common.population.to_string()),
        Blue.paint(common.iterations.to_string()),
        Blue.paint(common.evaluations.to_string()),
    );

    // Run algorithm
    let (_, evaluations) = match run_subcommand {
        &AlgorithmType::Single(run) => {
            let single_test_function = get_single(test_function)?;
            let fitness_evaluator =
                FitnessEvaluator::new(single_test_function, common.evaluations, &sampler);
            (
                run(&common, &fitness_evaluator, sub_m.unwrap()),
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
                run(&common, &fitness_evaluator, sub_m.unwrap()),
                fitness_evaluator.evaluations(),
            )
        }
    };

    let solutions = sampler.solutions();
    sampler.print_statistics(stdout());

    println!(
        "Number of fitness evaluations: {}",
        Green.paint(evaluations.to_string())
    );
    if let Objective::Single = sampler.objective {
        let best_solution = solutions
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
        if let Some(solution) = best_solution {
            println!(
                "Best solution: {} with fitness {}",
                Yellow.paint(format!("{:?}", solution.x)),
                Yellow.paint(format!("{:?}", solution.fitness))
            );
        }
    }
    write_solutions(
        "solutions.json",
        solutions,
        test_function_name,
        matches.is_present("plot-bounds"),
        common.upper_bound,
        common.lower_bound,
    );
    Ok(())
}

fn main() {
    if let Err(e) = start_algorithm() {
        eprintln!("{}: {}", Red.paint("Error"), e);
        process::exit(1);
    }
}
