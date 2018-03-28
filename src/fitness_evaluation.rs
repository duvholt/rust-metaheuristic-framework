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
}

impl<F> FitnessEvaluator<F> {
    pub fn new(test_function: fn(&Vec<f64>) -> F) -> FitnessEvaluator<F> {
        FitnessEvaluator { test_function }
    }

    pub fn calculate_fitness(&self, position: &Vec<f64>) -> F {
        (self.test_function)(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions::multi_dummy;
    use test_functions::sphere;

    #[test]
    fn fitness_evalator_calculates_single_fitness() {
        let fitness_evalator = FitnessEvaluator::new(sphere);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, 0.0);
    }

    #[test]
    fn fitness_evalator_calculates_multi_fitness() {
        let fitness_evalator = FitnessEvaluator::new(multi_dummy);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, vec![0.0, 0.0]);
    }
}
