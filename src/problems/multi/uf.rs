use fitness_evaluation::TestFunctionVar;
use std::collections::HashMap;
use std::f64::consts;

pub fn add_test_functions(test_functions_map: &mut HashMap<&'static str, TestFunctionVar>) {
    let bounds = get_upper_bounds(1);
    test_functions_map.insert(
        "uf1",
        TestFunctionVar::Multi(uf1, "uf1-2d", bounds.0, bounds.1),
    );
}

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert("uf", vec!["uf1".to_string()]);
}

fn get_upper_bounds(i: i8) -> (Vec<f64>, Vec<f64>) {
    match i {
        1 => {
            let mut ub = vec![1.0];
            let mut lb = vec![0.0];
            for _ in 1..30 {
                ub.push(1.0);
                lb.push(-1.0);
            }
            (lb, ub)
        }
        _ => panic!("Test function does not exist"),
    }
}

pub fn uf1(x: &Vec<f64>) -> Vec<f64> {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let mut even = 0.0;
    let mut odd = 0.0;

    for i in 1..x.len() {
        let a = (x[i] - (6.0 * consts::PI * x[0] + ((i + 1) as f64 * consts::PI) / x.len() as f64)
            .sin())
            .powi(2);
        if (i + 1) % 2 == 1 {
            f1 += a;
            odd += 1.0;
        } else {
            f2 += a;
            even += 1.0;
        }
    }

    f1 = x[0] + 2.0 / odd * f1;
    f2 = 1.0 - x[0].sqrt() + 2.0 / even * f2;
    vec![f1, f2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{weak_rng, Rng};

    #[test]
    fn uf1_optimum() {
        let mut rng = weak_rng();
        let dimensions = 30.0;
        let x1 = rng.gen_range(0.0, 1.0);
        let mut input = vec![x1];
        for j in 1..dimensions as usize {
            input.push((6.0 * consts::PI * x1 + ((j + 1) as f64 * consts::PI) / dimensions).sin());
        }
        let result = uf1(&input);
        println!("Input: {:?}", input);
        println!("Result: {:?}", result);
        assert!(result[0] >= 0.0 && result[0] <= 1.0 && result[1] == 1.0 - result[0].sqrt());
    }

    #[test]
    fn uf1_not_optimum() {
        let mut rng = weak_rng();
        let mut input = vec![rng.gen_range(-1.0, 1.0); 30];
        input[0] = 1.5;
        let result = uf1(&input);
        assert!(result[0] < 0.0 || result[0] > 1.0 || result[1] != 1.0 - result[0].sqrt());
    }
}
