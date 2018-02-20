extern crate rustoa;
#[macro_use]
extern crate clap;

use rustoa::Config;
use clap::{Arg, App};

fn main() {
    let matches = App::new("Simple Simulated Annealing implementation in Rust using Rosenbrock")
        .arg(Arg::with_name("start_t")
            .short("t")
            .long("temperature")
            .value_name("start_t")
            .help("Start temperature")
            .takes_value(true))
        .arg(Arg::with_name("cooldown")
            .short("-c")
            .long("cooldown")
            .value_name("cooldown")
            .help("Cooldown rate")
            .takes_value(true))
        .arg(Arg::with_name("iterations")
            .short("i")
            .long("iterations")
            .value_name("iterations")
            .help("Max iterations")
            .takes_value(true))
        .arg(Arg::with_name("space")
            .short("s")
            .long("space")
            .value_name("space")
            .help("Solution space size")
            .takes_value(true))
        .get_matches();

    let start_t = value_t!(matches, "start_t", f64).unwrap_or(1.0);
    let cooldown = value_t!(matches, "cooldown", f64).unwrap_or(0.9);
    let iterations = value_t!(matches, "iterations", i64).unwrap_or(1000);
    let space = value_t!(matches, "space", f64).unwrap_or(4.0);

    println!("Start T: {}, Cooldown: {}, Max iterations: {}, Space: {}", start_t, cooldown, iterations, space);

    let config = Config::new(start_t, cooldown, iterations, space);
    let solution = rustoa::run(config);

    println!("Final solution: ({:.2}, {:.2}) {}", solution.x, solution.y, solution.fitness);    
}
