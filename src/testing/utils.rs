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
    FitnessEvaluator::new(single_dummy, 10000, &sampler)
}

pub fn create_evaluator_multi(sampler: &Sampler) -> FitnessEvaluator<Vec<f64>> {
    FitnessEvaluator::new(multi_dummy, 10000, &sampler)
}

pub fn create_rng() -> StdRng {
    let seed: &[_] = &[1, 2, 3, 4];
    SeedableRng::from_seed(seed)
}

#[cfg(test)]
pub fn jmetal_compare(number: i8, uf: &Fn(&Vec<f64>) -> Vec<f64>, name: &str) {
    use serde_json;
    use std::fs::File;
    let file = File::open(format!(
        "jmetal_data/{}/variables-{}{}.json",
        name, name, number
    )).unwrap();
    let data: Vec<Vec<f64>> = serde_json::from_reader(file).unwrap();
    let file = File::open(format!(
        "jmetal_data/{}/fitness-{}{}.json",
        name, name, number
    )).unwrap();
    let fitness: Vec<Vec<f64>> = serde_json::from_reader(file).unwrap();
    for i in 0..data.len() {
        let test = uf(&data[i]);
        for j in 0..fitness[i].len() {
            assert_approx_eq!(test[j], fitness[i][j]);
        }
    }
}
