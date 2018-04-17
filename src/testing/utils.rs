use fitness_evaluation::FitnessEvaluator;
use rand::{SeedableRng, StdRng};
use solution::Objective;
use statistics::sampler::{Sampler, SamplerMode};
use testing::test_functions::{multi_dummy, single_dummy};

pub fn create_sampler() -> Sampler {
    Sampler::new(10, 10, SamplerMode::Evolution, Objective::Single)
}

pub fn create_sampler_multi() -> Sampler {
    Sampler::new(10, 10, SamplerMode::Evolution, Objective::Multi)
}

pub fn create_evaluator(sampler: &Sampler) -> FitnessEvaluator<f64> {
    FitnessEvaluator::new(single_dummy, 100, &sampler)
}

pub fn create_evaluator_multi(sampler: &Sampler) -> FitnessEvaluator<Vec<f64>> {
    FitnessEvaluator::new(multi_dummy, 100, &sampler)
}

pub fn create_rng() -> StdRng {
    let seed: &[_] = &[1, 2, 3, 4];
    SeedableRng::from_seed(seed)
}
