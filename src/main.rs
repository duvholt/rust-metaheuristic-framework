extern crate rustoa;

use std::env;
use std::process;

use rustoa::Config;

fn main() {
    let config = Config::env(env::args()).unwrap_or_else(|err| {
        eprintln!("Wowow! {}", err);
        process::exit(1);
    });

    let solution = rustoa::run(config);

    println!("Final solution: ({:.2}, {:.2}) {}", solution.x, solution.y, solution.fitness);    
}
