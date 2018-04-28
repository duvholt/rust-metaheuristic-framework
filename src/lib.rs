#![feature(test)]
extern crate ansi_term;
#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;
#[macro_use]
extern crate clap;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate itertools;
extern crate ordered_float;
extern crate serde_json;
extern crate statistical;
extern crate test;

pub mod algorithms;
pub mod archive;
pub mod config;
pub mod crossover;
pub mod distribution;
pub mod domination;
pub mod fitness_evaluation;
pub mod multiobjective;
pub mod position;
pub mod problems;
pub mod selection;
pub mod solution;
pub mod statistics;
pub mod testing;
