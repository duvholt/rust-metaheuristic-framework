use fitness_evaluation::FitnessEvaluator;
use solution::Objective;
use statistics::sampler::{Sampler, SamplerMode};
use test_functions::{multi_dummy, rosenbrock};

pub fn create_sampler() -> Sampler {
    Sampler::new(10, 10, SamplerMode::Evolution, Objective::Single)
}

pub fn create_sampler_multi() -> Sampler {
    Sampler::new(10, 10, SamplerMode::Evolution, Objective::Multi)
}

pub fn create_evaluator(sampler: &Sampler) -> FitnessEvaluator<f64> {
    FitnessEvaluator::new(rosenbrock, 100, &sampler)
}

pub fn create_evaluator_multi(sampler: &Sampler) -> FitnessEvaluator<Vec<f64>> {
    FitnessEvaluator::new(multi_dummy, 100, &sampler)
}
