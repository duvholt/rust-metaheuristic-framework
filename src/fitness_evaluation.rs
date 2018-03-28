use std::cell::Cell;

pub type SingleTestFunction = Fn(&Vec<f64>) -> f64;
pub type MultiTestFunction = Fn(&Vec<f64>) -> Vec<f64>;

pub type SingleTestFunctionVar = fn(&Vec<f64>) -> f64;
pub type MultiTestFunctionVar = fn(&Vec<f64>) -> Vec<f64>;
#[derive(Clone)]
pub enum TestFunctionVar {
    Single(SingleTestFunctionVar),
    Multi(MultiTestFunctionVar),
}

pub fn get_single(test_function_var: TestFunctionVar) -> SingleTestFunctionVar {
    match test_function_var {
        TestFunctionVar::Single(f) => f,
        _ => panic!("Algorithm only supports single objective functions"),
    }
}

pub fn get_multi(test_function_var: TestFunctionVar) -> MultiTestFunctionVar {
    match test_function_var {
        TestFunctionVar::Multi(f) => f,
        _ => panic!("Algorithm only supports multi objective functions"),
    }
}

pub struct FitnessEvaluator<F> {
    test_function: fn(&Vec<f64>) -> F,
    evaluations: Cell<i64>,
    max_evaluations: i64,
}

impl<F> FitnessEvaluator<F> {
    pub fn new(test_function: fn(&Vec<f64>) -> F, max_evaluations: i64) -> FitnessEvaluator<F> {
        FitnessEvaluator {
            test_function,
            evaluations: Cell::new(0),
            max_evaluations,
        }
    }

    pub fn calculate_fitness(&self, position: &Vec<f64>) -> F {
        self.evaluations.set(self.evaluations.get() + 1);
        (self.test_function)(position)
    }

    pub fn end_criteria(&self) -> bool {
        self.evaluations.get() >= self.max_evaluations
    }

    pub fn evaluations(&self) -> i64 {
        self.evaluations.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions::multi_dummy;
    use test_functions::sphere;

    #[test]
    fn fitness_evalator_calculates_single_fitness() {
        let fitness_evalator = FitnessEvaluator::new(sphere, 100);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, 0.0);
    }

    #[test]
    fn fitness_evalator_calculates_multi_fitness() {
        let fitness_evalator = FitnessEvaluator::new(multi_dummy, 100);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, vec![0.0, 0.0]);
    }

    #[test]
    fn fitness_evalator_end_criteria() {
        let fitness_evalator = FitnessEvaluator::new(multi_dummy, 5);

        assert_eq!(fitness_evalator.end_criteria(), false);
        for assert_value in vec![false, false, false, false, true, true, true] {
            fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);
            assert_eq!(fitness_evalator.end_criteria(), assert_value);
        }

        assert_eq!(fitness_evalator.evaluations(), 7);
    }
}
