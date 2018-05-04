use fitness_evaluation::TestFunctionVar;
use std::collections::HashMap;
use std::f64::consts;

pub fn add_test_functions(test_functions_map: &mut HashMap<&'static str, TestFunctionVar>) {
    test_functions_map.insert(
        "zdt1",
        TestFunctionVar::Multi(zdt1, "zdt1-2d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "zdt2",
        TestFunctionVar::Multi(zdt2, "zdt2-2d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "zdt3",
        TestFunctionVar::Multi(zdt3, "zdt3-2d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "zdt6",
        TestFunctionVar::Multi(zdt6, "zdt6-2d", vec![0.0; 30], vec![1.0; 30]),
    );
}

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert(
        "zdt",
        vec![
            "zdt1".to_string(),
            "zdt2".to_string(),
            "zdt3".to_string(),
            "zdt6".to_string(),
        ],
    );
}

pub fn zdt1(x: &Vec<f64>) -> Vec<f64> {
    let f1 = x[0];
    let sum: f64 = x.iter().skip(1).sum();
    let g = 1.0 + (9.0 / (x.len() as f64 - 1.0)) * sum;
    let h = 1.0 - (f1 / g).sqrt();
    let f2 = g * h;
    vec![f1, f2]
}

pub fn zdt2(x: &Vec<f64>) -> Vec<f64> {
    let f1 = x[0];
    let sum: f64 = x.iter().skip(1).sum();
    let g = 1.0 + (9.0 / (x.len() as f64 - 1.0)) * sum;
    let h = 1.0 - (f1 / g).powi(2);
    let f2 = g * h;
    vec![f1, f2]
}

pub fn zdt3(x: &Vec<f64>) -> Vec<f64> {
    let f1 = x[0];
    let sum: f64 = x.iter().skip(1).sum();
    let g = 1.0 + (9.0 / (x.len() as f64 - 1.0)) * sum;
    let h = 1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * consts::PI * f1).sin();
    let f2 = g * h;
    vec![f1, f2]
}

pub fn zdt6(x: &Vec<f64>) -> Vec<f64> {
    let f1 = 1.0 - (-4.0 * x[0]).exp() * (6.0 * consts::PI * x[0]).sin().powi(6);
    let sum: f64 = x.iter().skip(1).sum();
    let g = 1.0 + 9.0 * (sum / 9.0).powf(0.25);
    let h = 1.0 - (f1 / g).powi(2);
    let f2 = g * h;
    vec![f1, f2]
}
