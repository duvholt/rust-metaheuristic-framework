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
    let bounds = zdt4_bounds();
    test_functions_map.insert(
        "zdt4",
        TestFunctionVar::Multi(zdt4, "zdt4-2d", bounds.0, bounds.1),
    );
    test_functions_map.insert(
        "zdt6",
        TestFunctionVar::Multi(zdt6, "zdt6-2d", vec![0.0; 30], vec![1.0; 30]),
    );
}

pub fn zdt4_bounds() -> (Vec<f64>, Vec<f64>) {
    let mut upper_bound = vec![1.0];
    let mut lower_bound = vec![0.0];
    for _ in 0..29 {
        upper_bound.push(5.0);
        lower_bound.push(-5.0);
    }
    (lower_bound, upper_bound)
}

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert(
        "zdt",
        vec![
            "zdt1".to_string(),
            "zdt2".to_string(),
            "zdt3".to_string(),
            "zdt4".to_string(),
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

pub fn zdt4(x: &Vec<f64>) -> Vec<f64> {
    let g = 1.0 + 10.0 * (x.len() as f64 - 1.0)
        + (1..x.len())
            .map(|i| x[i].powi(2) - 10.0 * (4.0 * consts::PI * x[i]).cos())
            .sum::<f64>();
    let f1 = x[0];
    let f2 = g * (1.0 - (x[0] / g).sqrt());
    vec![f1, f2]
}

pub fn zdt6(x: &Vec<f64>) -> Vec<f64> {
    let f1 = 1.0 - (-4.0 * x[0]).exp() * (6.0 * consts::PI * x[0]).sin().powi(6);
    let sum: f64 = x.iter().skip(1).sum();
    let g = 1.0 + 9.0 * (sum / (x.len() - 1) as f64).powf(0.25);
    let h = 1.0 - (f1 / g).powi(2);
    let f2 = g * h;
    vec![f1, f2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::utils::jmetal_compare;

    #[test]
    fn zdt1_jmetal_compare() {
        jmetal_compare(1, &zdt1, "zdt");
    }

    #[test]
    fn zdt2_jmetal_compare() {
        jmetal_compare(2, &zdt2, "zdt");
    }

    #[test]
    fn zdt3_jmetal_compare() {
        jmetal_compare(3, &zdt3, "zdt");
    }

    #[test]
    fn zdt4_jmetal_compare() {
        jmetal_compare(4, &zdt4, "zdt");
    }

    #[test]
    fn zdt6_jmetal_compare() {
        jmetal_compare(6, &zdt6, "zdt");
    }
}
