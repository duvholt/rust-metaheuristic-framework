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
