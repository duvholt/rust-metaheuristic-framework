#[macro_use]
extern crate clap;
extern crate rustoa;
extern crate serde_json;

use rustoa::test_functions;
use test_functions::TestFunctionVar;
use rustoa::algorithms::sa;
use rustoa::algorithms::dummy;
use rustoa::algorithms::pso;
use rustoa::algorithms::ewa;
use rustoa::algorithms::mopso;
use rustoa::solution::{SolutionJSON, Solutions};
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
                .possible_values(&["ackley", "himmelblau", "rosenbrock", "zakharov", "zdt1", "schaffer1"])
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
            Arg::with_name("space")
                .short("s")
                .long("space")
                .value_name("space")
                .help("Solution space size")
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
        .subcommand(
            SubCommand::with_name("sa")
                .about("simulated annealing")
                .arg(
                    Arg::with_name("start_t")
                        .short("t")
                        .long("temperature")
                        .value_name("start_t")
                        .help("Start temperature")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("cooldown")
                        .short("-c")
                        .long("cooldown")
                        .value_name("cooldown")
                        .help("Cooldown rate")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("dummy").about("dummy solver").arg(
                Arg::with_name("example")
                    .long("example")
                    .value_name("example")
                    .help("example argument")
                    .takes_value(true),
            ),
        )
        .subcommand(
            SubCommand::with_name("pso")
                .about("particle swarm optimization")
                .arg(
                    Arg::with_name("c1")
                        .long("c1")
                        .value_name("c1")
                        .help("C1 constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("c2")
                        .long("c2")
                        .value_name("c2")
                        .help("C2 constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("inertia")
                        .long("inertia")
                        .value_name("inertia")
                        .help("inertia constant")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("mopso")
                .about("particle swarm optimization")
                .arg(
                    Arg::with_name("c1")
                        .long("c1")
                        .value_name("c1")
                        .help("C1 constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("c2")
                        .long("c2")
                        .value_name("c2")
                        .help("C2 constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("inertia")
                        .long("inertia")
                        .value_name("inertia")
                        .help("inertia constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("archive_size")
                        .short("-a")
                        .long("archive_size")
                        .value_name("archive_size")
                        .help("archive size")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("divisions")
                        .long("divisions")
                        .value_name("divisions")
                        .help("number of archive divisions")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("mutation_rate")
                        .short("-m")
                        .long("mutation_rate")
                        .value_name("mutation_rate")
                        .help("mutation rate")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("ewa")
                .about("earth worm optimization algorithm")
                .arg(
                    Arg::with_name("beta")
                        .long("beta")
                        .value_name("beta")
                        .help("beta constant")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("similarity")
                        .long("similarity")
                        .value_name("similarity")
                        .help("similarity constant")
                        .takes_value(true),
                ),
        )
        .get_matches();

    let verbose = matches.is_present("verbose");
    let iterations = value_t!(matches, "iterations", i64).unwrap_or(1000);
    let space = value_t!(matches, "space", f64).unwrap_or(4.0);
    let dimension = value_t!(matches, "dimension", usize).unwrap_or(2);
    let population = value_t!(matches, "population", usize).unwrap_or(50);
    let test_function_name = value_t!(matches, "test_function", String).unwrap();

    let test_function = match test_function_name.as_ref() {
        "rosenbrock" => TestFunctionVar::Single(test_functions::rosenbrock),
        "zakharov" => TestFunctionVar::Single(test_functions::zakharov),
        "ackley" => TestFunctionVar::Single(test_functions::ackley),
        "himmelblau" => TestFunctionVar::Single(test_functions::himmelblau),
        "zdt1" => TestFunctionVar::Multi(test_functions::zdt1),
        "schaffer1" => TestFunctionVar::Multi(test_functions::schaffer1),
        _ => panic!("Test function does not exist"),
    };

    println!(
        "Max iterations: {}, Space: {}, Function: {}, Population: {}",
        iterations, space, test_function_name, population
    );

    let solutions = match matches.subcommand() {
        ("sa", Some(sub_m)) => {
            let start_t = value_t!(sub_m, "start_t", f64).unwrap_or(1.0);
            let cooldown = value_t!(sub_m, "cooldown", f64).unwrap_or(0.9);

            println!(
                "Running SA with start T: {}, cooldown: {}",
                start_t, cooldown
            );
            let config = sa::Config::new(start_t, cooldown, iterations, space, dimension);

            sa::run(config, &test_functions::get_single(test_function))
        }
        ("pso", Some(sub_m)) => {
            let c1 = value_t!(sub_m, "c1", f64).unwrap_or(2.0);
            let c2 = value_t!(sub_m, "c2", f64).unwrap_or(2.0);
            let inertia = value_t!(sub_m, "inertia", f64).unwrap_or(1.1);
            println!(
                "Running PSO with C1: {}, C2: {} inertia: {}",
                c1, c2, inertia
            );

            let config = pso::Config {
                space,
                dimension,
                iterations,
                population,
                c1,
                c2,
                inertia,
            };
            pso::run(config, &test_functions::get_single(test_function))
        }
        ("mopso", Some(sub_m)) => {
            let c1 = value_t!(sub_m, "c1", f64).unwrap_or(1.0);
            let c2 = value_t!(sub_m, "c2", f64).unwrap_or(2.0);
            let inertia = value_t!(sub_m, "inertia", f64).unwrap_or(0.4);
            let archive_size = value_t!(sub_m, "archive_size", usize).unwrap_or(population);
            let divisions = value_t!(sub_m, "divisions", usize).unwrap_or(30);
            let mutation_rate = value_t!(sub_m, "mutation_rate", f64).unwrap_or(0.1);
            println!(
                "Running MOPSO with C1: {}, C2: {} inertia: {}",
                c1, c2, inertia
            );

            let config = mopso::Config {
                lower_space: 0.0,
                upper_space: 1.0,
                dimension,
                iterations,
                population,
                c1,
                c2,
                inertia,
                archive_size,
                divisions,
                mutation_rate,
                verbose,
            };
            mopso::run(config, &test_functions::get_multi(test_function))
        }
        ("ewa", Some(sub_m)) => {
            let beta = value_t!(sub_m, "beta", f64).unwrap_or(1.0);
            let similarity = value_t!(sub_m, "similarity", f64).unwrap_or(0.98);
            println!("Running EWA with beta: {} similarity: {}", beta, similarity);

            let config = ewa::Config {
                space,
                dimension,
                iterations,
                population,
                beta,
                similarity,
            };
            ewa::run(config, &test_functions::get_single(test_function))
        }
        ("dummy", Some(sub_m)) => {
            let example = value_t!(sub_m, "example", f64).unwrap_or(1.0);
            println!("Running dummy solver with example: {}", example);
            let config = dummy::Config::new(example);

            dummy::run(config, &test_functions::get_single(test_function))
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
