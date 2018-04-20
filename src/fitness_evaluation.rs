use statistics::sampler::Sampler;
use std::cell::Cell;

pub type SingleTestFunction = Fn(&Vec<f64>) -> f64;
pub type MultiTestFunction = Fn(&Vec<f64>) -> Vec<f64>;

pub type SingleTestFunctionVar = fn(&Vec<f64>) -> f64;
pub type MultiTestFunctionVar = fn(&Vec<f64>) -> Vec<f64>;
#[derive(Clone, Copy)]
pub enum TestFunctionVar {
    Single(SingleTestFunctionVar),
    Multi(MultiTestFunctionVar, &'static str),
}

pub fn get_single(
    test_function_var: TestFunctionVar,
) -> Result<SingleTestFunctionVar, &'static str> {
    match test_function_var {
        TestFunctionVar::Single(f) => Ok(f),
        _ => Err("Algorithm only supports single objective functions"),
    }
}

pub fn get_multi(
    test_function_var: TestFunctionVar,
) -> Result<(MultiTestFunctionVar, &'static str), &'static str> {
    match test_function_var {
        TestFunctionVar::Multi(f, s) => Ok((f, s)),
        _ => Err("Algorithm only supports multi objective functions"),
    }
}

pub struct FitnessEvaluator<'a, F: 'static>
where
    F: Clone,
{
    test_function: fn(&Vec<f64>) -> F,
    evaluations: Cell<i64>,
    max_evaluations: i64,
    pub sampler: &'a Sampler,
}

impl<'a, F: 'static> FitnessEvaluator<'a, F>
where
    F: Clone,
{
    pub fn new(
        test_function: fn(&Vec<f64>) -> F,
        max_evaluations: i64,
        sampler: &'a Sampler,
    ) -> FitnessEvaluator<F> {
        FitnessEvaluator {
            test_function,
            evaluations: Cell::new(0),
            max_evaluations,
            sampler,
        }
    }

    pub fn end_criteria(&self) -> bool {
        self.evaluations.get() >= self.max_evaluations
    }

    pub fn evaluations(&self) -> i64 {
        self.evaluations.get()
    }
}

impl<'a> FitnessEvaluator<'a, f64> {
    pub fn calculate_fitness(&self, position: &Vec<f64>) -> f64 {
        self.evaluations.set(self.evaluations.get() + 1);
        let fitness = (self.test_function)(position);
        self.sampler.sample_fitness_single(&fitness, &position);
        fitness
    }
}

impl<'a> FitnessEvaluator<'a, Vec<f64>> {
    pub fn calculate_fitness(&self, position: &Vec<f64>) -> Vec<f64> {
        self.evaluations.set(self.evaluations.get() + 1);
        let fitness = (self.test_function)(position);
        self.sampler.sample_fitness_multi(&fitness, &position);
        fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::test_functions::multi_dummy;
    use testing::test_functions::single_dummy;
    use testing::utils::create_sampler_multi;

    #[test]
    fn fitness_evalator_calculates_single_fitness() {
        let sampler = create_sampler_multi();
        let fitness_evalator = FitnessEvaluator::new(single_dummy, 100, &sampler);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, 0.0);
    }

    #[test]
    fn fitness_evalator_calculates_multi_fitness() {
        let sampler = create_sampler_multi();
        let fitness_evalator = FitnessEvaluator::new(multi_dummy, 100, &sampler);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, vec![0.0, 0.0]);
    }

    #[test]
    fn fitness_evalator_end_criteria() {
        let sampler = create_sampler_multi();
        let fitness_evalator = FitnessEvaluator::new(multi_dummy, 5, &sampler);

        assert_eq!(fitness_evalator.end_criteria(), false);
        for assert_value in vec![false, false, false, false, true, true, true] {
            fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);
            assert_eq!(fitness_evalator.end_criteria(), assert_value);
        }

        assert_eq!(fitness_evalator.evaluations(), 7);
    }
}
