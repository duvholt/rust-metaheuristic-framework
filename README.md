# Metaheuristic Optimization Framework

[![CircleCI](https://circleci.com/gh/duvholt/multi-objective-random-animal.png?circle-token=84a65873909c24aa9488f284ec3ec9a0ad23ee19)](https://circleci.com/gh/duvholt/multi-objective-random-animal)
[![codecov](https://codecov.io/gh/duvholt/multi-objective-random-animal/branch/master/graph/badge.svg?token=OeoHWrHHqY)](https://codecov.io/gh/duvholt/multi-objective-random-animal)

This project consists of several algorithms for both single and multi-objective optimization.
The framework and optimization algorithm code is located in `src` while the plotting code is located in the `plotting folder`.

## Installation

The project has been developed using the nightly branch of Rust, but with the exception of the built-in benchmarker it should work with the stable branch.

Install Rust using https://www.rustup.rs/

Build the project using `cargo build --release` or run it directly with `cargo run --release`. 

## Run

The project comes with a CLI. Check the help output for a detailed list of all the options:
`cargo run --release -- --help`

Example: `cargo run --release -- -f uf3 zdt3 -p 100 moamo`

## Formatting

The code is formatted using [rustfmt](https://github.com/rust-lang-nursery/rustfmt)

## Testing

Run tests using `cargo test`

Run benchmarks using `cargo bench -- --ignored` (requires nightly at the time of writing).
All benchmarks should be ignored to avoid contributing to coverage

## Plotting

Plots are generated using Python 3 with Matplotlib.
Data is read from solutions.json in the main directory. Generate solutions by using the Rust project.

```bash
cd plotting
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
python plot.py
```
