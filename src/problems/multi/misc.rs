use fitness_evaluation::TestFunctionVar;
use std::collections::HashMap;

pub fn add_test_functions(test_functions_map: &mut HashMap<&'static str, TestFunctionVar>) {
    test_functions_map.insert(
        "schaffer1",
        TestFunctionVar::Multi(schaffer1, "schaffer1-2d"),
    );
}

pub fn schaffer1(x: &Vec<f64>) -> Vec<f64> {
    vec![x[0].powf(2.0), (x[0] - 2.0).powf(2.0)]
}
