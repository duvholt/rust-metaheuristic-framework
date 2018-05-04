use fitness_evaluation::TestFunctionVar;
use std::collections::HashMap;
use std::f64::consts;

pub fn add_test_functions(test_functions_map: &mut HashMap<&'static str, TestFunctionVar>) {
    let bounds = get_upper_bounds(1);
    test_functions_map.insert(
        "uf1",
        TestFunctionVar::Multi(uf1, "uf1-2d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(2);
    test_functions_map.insert(
        "uf2",
        TestFunctionVar::Multi(uf2, "uf2-2d", bounds.0, bounds.1),
    );
}

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert("uf", vec!["uf1".to_string(), "uf2".to_string()]);
}

fn get_upper_bounds(i: i8) -> (Vec<f64>, Vec<f64>) {
    match i {
        1 | 2 => {
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

pub fn uf2(x: &Vec<f64>) -> Vec<f64> {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let mut even = 0.0;
    let mut odd = 0.0;

    for i in 1..x.len() {
        let a = 0.3 * x[0].powi(2)
            * (24.0 * consts::PI * x[0] + (4.0 * (i + 1) as f64 * consts::PI) / x.len() as f64)
                .cos() + 0.6 * x[0];
        let b = 6.0 * consts::PI * x[0] + ((i + 1) as f64 * consts::PI) / x.len() as f64;
        if (i + 1) % 2 == 1 {
            f1 += (x[i] - a * b.cos()).powi(2);
            odd += 1.0;
        } else {
            f2 += (x[i] - a * b.sin()).powi(2);
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
        for i in 1..dimensions as usize {
            input.push((6.0 * consts::PI * x1 + ((i + 1) as f64 * consts::PI) / dimensions).sin());
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

    #[test]
    fn uf2_optimum() {
        let mut rng = weak_rng();
        let dimensions = 30.0;
        let x1 = rng.gen_range(0.0, 1.0) as f64;
        let mut input = vec![x1];
        for i in 1..dimensions as usize {
            let a = 0.3 * x1.powi(2)
                * (24.0 * consts::PI * x1 + (4.0 * (i + 1) as f64 * consts::PI) / dimensions).cos()
                + 0.6 * x1;
            let b = 6.0 * consts::PI * x1 + ((i + 1) as f64 * consts::PI) / dimensions;
            if (i + 1) % 2 == 1 {
                input.push(a * b.cos());
            } else {
                input.push(a * b.sin());
            }
        }
        let result = uf2(&input);
        println!("Input: {:?}", input);
        println!("Result: {:?}", result);
        assert!(result[0] >= 0.0 && result[0] <= 1.0 && result[1] == 1.0 - result[0].sqrt());
    }

    #[test]
    fn uf2_not_optimum() {
        let mut rng = weak_rng();
        let mut input = vec![rng.gen_range(-1.0, 1.0); 30];
        input[0] = 1.5;
        let result = uf2(&input);
        assert!(result[0] < 0.0 || result[0] > 1.0 || result[1] != 1.0 - result[0].sqrt());
    }
}
