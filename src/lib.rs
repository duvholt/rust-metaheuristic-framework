#![feature(test)]
#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate test;

pub mod test_functions;
pub mod algorithms;
pub mod solution;
