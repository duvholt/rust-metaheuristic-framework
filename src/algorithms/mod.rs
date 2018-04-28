use clap::{App, ArgMatches};
use config::CommonConfig;
use fitness_evaluation::FitnessEvaluator;
use solution::SolutionJSON;
use std::collections::HashMap;

pub mod amo;
pub mod da;
pub mod dummy;
pub mod ewa;
pub mod loa;
pub mod mopso;
pub mod pso;
pub mod sa;

pub type AlgorithmSubCommand = fn(&str) -> App<'static, 'static>;
pub type AlgorithmRun<S> =
    fn(&CommonConfig, &FitnessEvaluator<S>, &ArgMatches) -> Vec<SolutionJSON>;
pub enum AlgorithmType {
    Single(AlgorithmRun<f64>),
    Multi(AlgorithmRun<Vec<f64>>),
}

pub fn add_algorithms(algorithms: &mut HashMap<&str, (AlgorithmSubCommand, AlgorithmType)>) {
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
}
