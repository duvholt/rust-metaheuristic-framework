# Multi Objective Algorithm

[![CircleCI](https://circleci.com/gh/duvholt/multi-objective-random-animal.svg?style=svg)](https://circleci.com/gh/duvholt/multi-objective-random-animal)

Programming languages:

- Rust (algorithm)
- Python 3 (plotting)

## Algorithm

The project consists of several optimization algorithms both single and multi objective. 

### Installation

The project has been developed using the nightly branch of Rust, but with the exception of the built-in benchmarker it should work with the stable branch.

Install Rust using https://www.rustup.rs/

Build the project using `cargo build --release` or run it directly with `cargo run --release`. 

### Run

The project comes with a cli. See help for a detailed list of all the options.
`cargo run --release -- --help`

Example: `cargo run --release -- -f ackley -d 30 - pso`

### Formatting

The code is formatted using [rustfmt](https://github.com/rust-lang-nursery/rustfmt)

### Testing

Run tests using `cargo test`

Run benchmarks using `cargo bench` (requires nightly at the time of writing).

## Plotting

Plots are generated with Matplotlib. 
Data is read from solutions.json in the main directory. Generate solutions by using the Rust project.

```bash
cd plotting
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
python plot.py
# or
python multi_plot.py
```
