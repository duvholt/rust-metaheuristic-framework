extern crate ansi_term;
#[macro_use]
extern crate clap;
extern crate rustoa;
extern crate serde_json;

use ansi_term::Color::{Blue, Cyan, Green, Red};
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
use rustoa::testfunctions;
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
            .multiple(true)
            .takes_value(true),
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

fn run_algorithm(
    algorithm: &AlgorithmType,
    sub_m: &ArgMatches,
    test_function: TestFunctionVar,
    mut sampler: Sampler,
    common: &CommonConfig,
    number_of_runs: usize,
) -> Result<Vec<SolutionJSON>, &'static str> {
    for run in 0..number_of_runs {
        println!("Starting run #{}", Blue.paint(run.to_string()));

        // Run algorithm
        let (_, evaluations) = match algorithm {
            &AlgorithmType::Single(run) => {
                let single_test_function = get_single(test_function)?;
                let fitness_evaluator =
                    FitnessEvaluator::new(single_test_function, common.evaluations, &sampler);
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

        sampler.print_run_statistics(stdout());
        println!(
            "Number of fitness evaluations: {}",
            Green.paint(evaluations.to_string())
        );

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
        TestFunctionVar::Single(testfunctions::rosenbrock),
    );
    test_functions_map.insert("zakharov", TestFunctionVar::Single(testfunctions::zakharov));
    test_functions_map.insert("ackley", TestFunctionVar::Single(testfunctions::ackley));
    test_functions_map.insert(
        "himmelblau",
        TestFunctionVar::Single(testfunctions::himmelblau),
    );
    test_functions_map.insert("sphere", TestFunctionVar::Single(testfunctions::sphere));
    test_functions_map.insert(
        "rastrigin",
        TestFunctionVar::Single(testfunctions::rastrigin),
    );
    test_functions_map.insert(
        "hyper-ellipsoid",
        TestFunctionVar::Single(testfunctions::axis_parallel_hyper_ellipsoid),
    );
    test_functions_map.insert(
        "moved-hyper-ellipsoid",
        TestFunctionVar::Single(testfunctions::moved_axis_parallel_hyper_ellipsoid),
    );
    test_functions_map.insert(
        "high-elliptic",
        TestFunctionVar::Single(testfunctions::high_elliptic),
    );
    test_functions_map.insert(
        "bent-cigar",
        TestFunctionVar::Single(testfunctions::bent_cigar),
    );
    test_functions_map.insert("griewank", TestFunctionVar::Single(testfunctions::griewank));
    test_functions_map.insert("schwefel", TestFunctionVar::Single(testfunctions::schwefel));
    test_functions_map.insert("katsuura", TestFunctionVar::Single(testfunctions::katsuura));
    test_functions_map.insert(
        "weierstrass",
        TestFunctionVar::Single(testfunctions::weierstrass),
    );
    test_functions_map.insert("happycat", TestFunctionVar::Single(testfunctions::happycat));
    test_functions_map.insert("hgbat", TestFunctionVar::Single(testfunctions::hgbat));
    test_functions_map.insert("levy05", TestFunctionVar::Single(testfunctions::levy05));
    test_functions_map.insert("easom", TestFunctionVar::Single(testfunctions::easom));
    // Multi-objective
    test_functions_map.insert(
        "schaffer1",
        TestFunctionVar::Multi(testfunctions::schaffer1, "schaffer1-2d"),
    );
    test_functions_map.insert(
        "zdt1",
        TestFunctionVar::Multi(testfunctions::zdt1, "zdt1-2d"),
    );
    test_functions_map.insert(
        "zdt2",
        TestFunctionVar::Multi(testfunctions::zdt2, "zdt2-2d"),
    );
    test_functions_map.insert(
        "zdt3",
        TestFunctionVar::Multi(testfunctions::zdt3, "zdt3-2d"),
    );
    test_functions_map.insert(
        "zdt6",
        TestFunctionVar::Multi(testfunctions::zdt6, "zdt6-2d"),
    );
    test_functions_map.insert(
        "dtlz1",
        TestFunctionVar::Multi(testfunctions::dtlz1, "dtlz1-3d"),
    );
    test_functions_map.insert(
        "dtlz2",
        TestFunctionVar::Multi(testfunctions::dtlz2, "dtlz2-3d"),
    );
    test_functions_map.insert(
        "dtlz3",
        TestFunctionVar::Multi(testfunctions::dtlz3, "dtlz3-3d"),
    );
    test_functions_map.insert(
        "dtlz4",
        TestFunctionVar::Multi(testfunctions::dtlz4, "dtlz4-3d"),
    );
    test_functions_map.insert(
        "dtlz5",
        TestFunctionVar::Multi(testfunctions::dtlz5, "dtlz5-3d"),
    );
    test_functions_map.insert(
        "dtlz6",
        TestFunctionVar::Multi(testfunctions::dtlz6, "dtlz6-3d"),
    );
    test_functions_map.insert(
        "dtlz7",
        TestFunctionVar::Multi(testfunctions::dtlz7, "dtlz7-3d"),
    );

    let matches = arguments(&test_functions_map, &algorithms);

    // Test function
    let test_function_names = values_t_or_exit!(matches, "test_function", String);
    let number_of_runs = value_t_or_exit!(matches, "runs", usize);

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
        let solutions = run_algorithm(
            &run_subcommand,
            sub_m.unwrap(),
            test_function,
            sampler,
            &common,
            number_of_runs,
        )?;
        write_solutions(
            "solutions.json",
            solutions,
            test_function_name,
            plot_bounds,
            common.upper_bound,
            common.lower_bound,
        );
    }

    Ok(())
}

fn main() {
    if let Err(e) = start_algorithm() {
        eprintln!("{}: {}", Red.paint("Error"), e);
        process::exit(1);
    }
}
