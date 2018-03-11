#[macro_use]
extern crate clap;
extern crate rustoa;
extern crate serde_json;

use rustoa::test_functions;
use rustoa::algorithms::sa;
use rustoa::algorithms::dummy;
use rustoa::algorithms::pso;
use rustoa::algorithms::ewa;
use rustoa::solution::{Solution, Solutions};
use clap::{App, Arg, SubCommand};
use std::fs::File;
use std::io::prelude::*;

fn write_solutions(filename: &str, solutions: Vec<Solution>, test_function: String) {
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
                .possible_values(&["ackley", "himmelblau", "rosenbrock", "zakharov"])
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
            SubCommand::with_name("ewa")
                .about("earth worm optimization algorithm")
                .arg(
                    Arg::with_name("n_kew")
                        .long("n-kew")
                        .value_name("n_kew")
                        .help("non-kept earth worms")
                        .takes_value(true),
                )
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

    let iterations = value_t!(matches, "iterations", i64).unwrap_or(1000);
    let space = value_t!(matches, "space", f64).unwrap_or(4.0);
    let dimension = value_t!(matches, "dimension", i32).unwrap_or(2);
    let test_function_name = value_t!(matches, "test_function", String).unwrap();

    let test_function = match test_function_name.as_ref() {
        "rosenbrock" => test_functions::rosenbrock,
        "zakharov" => test_functions::zakharov,
        "ackley" => test_functions::ackley,
        "himmelblau" => test_functions::himmelblau,
        _ => panic!("Test function does not exist"),
    };

    println!(
        "Max iterations: {}, Space: {}, Function: {}",
        iterations, space, test_function_name
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

            sa::run(config, &test_function)
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
                c1: 0.3,
                c2: 0.5,
                inertia: 0.5,
            };
            pso::run(config, &test_function)
        }
        ("ewa", Some(sub_m)) => {
            let n_kew = value_t!(sub_m, "n_kew", usize).unwrap_or(50);
            let beta = value_t!(sub_m, "beta", f64).unwrap_or(1.0);
            let similarity = value_t!(sub_m, "similarity", f64).unwrap_or(0.98);
            println!(
                "Running EWA with n_kew: {}, beta: {} similarity: {}",
                n_kew, beta, similarity
            );

            let config = ewa::Config {
                space,
                dimension,
                iterations,
                population: 50,
                n_kew,
                beta,
                similarity,
            };
            ewa::run(config, &test_function)
        }
        ("dummy", Some(sub_m)) => {
            let example = value_t!(sub_m, "example", f64).unwrap_or(1.0);
            println!("Running dummy solver with example: {}", example);
            let config = dummy::Config::new(example);

            dummy::run(config, &test_function)
        }
        _ => {
            panic!("Algorithm was not specified!");
        }
    };

    if let Some(solution) = solutions.last() {
        println!("Final solution: ({:?}) {}", solution.x, solution.fitness);
    }

    write_solutions("solutions.json", solutions, test_function_name);
}
