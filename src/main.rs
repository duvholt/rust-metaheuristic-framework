#[macro_use]
extern crate clap;
extern crate rustoa;
extern crate serde_json;

use rustoa::test_functions;
use test_functions::TestFunctionVar;
use rustoa::algorithms::sa;
use rustoa::algorithms::da;
use rustoa::algorithms::dummy;
use rustoa::algorithms::pso;
use rustoa::algorithms::ewa;
use rustoa::algorithms::mopso;
use rustoa::solution::{SolutionJSON, Solutions};
use rustoa::config::CommonConfig;
use clap::{App, Arg, SubCommand};
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

fn main() {
    let matches = App::new("Simple Simulated Annealing implementation in Rust using Rosenbrock")
        .arg(
            Arg::with_name("test_function")
                .short("f")
                .long("test-function")
                .value_name("test_function")
                .help("Name of test function")
                .required(true)
                .possible_values(&[
                    "ackley",
                    "himmelblau",
                    "rosenbrock",
                    "zakharov",
                    "zdt1",
                    "schaffer1",
                    "hyper-ellipsoid",
                    "moved-hyper-ellipsoid",
                    "sphere",
                    "rastrigin",
                ])
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
        .subcommand(sa::subcommand("sa"))
        .subcommand(dummy::subcommand("dummy"))
        .subcommand(pso::subcommand("pso"))
        .subcommand(mopso::subcommand("mopso"))
        .subcommand(ewa::subcommand("ewa"))
        .subcommand(
            SubCommand::with_name("da")
                .about("Dandelion algorithm")
                .arg(
                    Arg::with_name("r")
                        .short("-r")
                        .long("r")
                        .value_name("r")
                        .help("r constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("e")
                        .short("-e")
                        .long("e")
                        .value_name("e")
                        .help("e constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("normal_seeds")
                        .short("-n")
                        .long("normal_seeds")
                        .value_name("normal_seeds")
                        .help("normal_seeds constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("self_learning_seeds")
                        .short("-s")
                        .long("self_learning_seeds")
                        .value_name("self_learning_seeds")
                        .help("self_learning_seeds constant")
                        .takes_value(true),
                ),
        )
        .get_matches();

    let test_function_name = value_t!(matches, "test_function", String).unwrap();
    let test_function = match test_function_name.as_ref() {
        "rosenbrock" => TestFunctionVar::Single(test_functions::rosenbrock),
        "zakharov" => TestFunctionVar::Single(test_functions::zakharov),
        "ackley" => TestFunctionVar::Single(test_functions::ackley),
        "himmelblau" => TestFunctionVar::Single(test_functions::himmelblau),
        "sphere" => TestFunctionVar::Single(test_functions::sphere),
        "rastrigin" => TestFunctionVar::Single(test_functions::rastrigin),
        "zdt1" => TestFunctionVar::Multi(test_functions::zdt1),
        "schaffer1" => TestFunctionVar::Multi(test_functions::schaffer1),
        "hyper-ellipsoid" => TestFunctionVar::Single(test_functions::axis_parallel_hyper_ellipsoid),
        "moved-hyper-ellipsoid" => {
            TestFunctionVar::Single(test_functions::moved_axis_parallel_hyper_ellipsoid)
        }
        _ => panic!("Test function does not exist"),
    };

    let upper_bound = value_t!(matches, "upper_bound", f64).unwrap_or(4.0);
    let common = CommonConfig {
        verbose: matches.is_present("verbose"),
        iterations: value_t!(matches, "iterations", i64).unwrap_or(1000),
        upper_bound,
        lower_bound: value_t!(matches, "lower_bound", f64).unwrap_or(-upper_bound),
        dimension: value_t!(matches, "dimension", usize).unwrap_or(2),
        population: value_t!(matches, "population", usize).unwrap_or(50),
    };

    println!(
        "Max iterations: {}, Upper bound: {}, Lower bound: {}, Function: {}, Population: {}",
        common.iterations,
        common.upper_bound,
        common.lower_bound,
        test_function_name,
        common.population
    );

    let solutions = match matches.subcommand() {
        ("sa", Some(sub_m)) => {
            sa::run_subcommand(&common, test_functions::get_single(test_function), sub_m)
        }
        ("pso", Some(sub_m)) => {
            pso::run_subcommand(&common, test_functions::get_single(test_function), sub_m)
        }
        ("mopso", Some(sub_m)) => {
            mopso::run_subcommand(&common, test_functions::get_multi(test_function), sub_m)
        }
        ("ewa", Some(sub_m)) => {
            ewa::run_subcommand(&common, test_functions::get_single(test_function), sub_m)
        }
        ("dummy", Some(sub_m)) => {
            dummy::run_subcommand(&common, test_functions::get_single(test_function), sub_m)
        }
        ("da", Some(sub_m)) => {
            let r = value_t!(sub_m, "r", f64).unwrap_or(0.95);
            let e = value_t!(sub_m, "e", f64).unwrap_or(1.05);
            let normal_seeds = value_t!(sub_m, "normal_seeds", i64).unwrap_or(200);
            let self_learning_seeds = value_t!(sub_m, "self_learning_seeds", i64).unwrap_or(10);
            println!(
                "Running DA with r: {} e: {} normal_seeds: {} self_learning_seeds: {}",
                r, e, normal_seeds, self_learning_seeds
            );
            let config = da::Config {
                upper_bound: common.upper_bound,
                lower_bound: common.lower_bound,
                dimension: common.dimension,
                iterations: common.iterations,
                population: common.population,
                r,
                e,
                normal_seeds,
                self_learning_seeds,
            };

            da::run(config, &test_functions::get_single(test_function))
        }
        _ => {
            panic!("Algorithm was not specified!");
        }
    };

    if let Some(solution) = solutions.last() {
        println!("Final solution: ({:?}) {:?}", solution.x, solution.fitness);
    }

    write_solutions("solutions.json", solutions, test_function_name);
}
