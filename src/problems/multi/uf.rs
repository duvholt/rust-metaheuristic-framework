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
    let bounds = get_upper_bounds(3);
    test_functions_map.insert(
        "uf3",
        TestFunctionVar::Multi(uf3, "uf3-2d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(4);
    test_functions_map.insert(
        "uf4",
        TestFunctionVar::Multi(uf4, "uf4-2d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(5);
    test_functions_map.insert(
        "uf5",
        TestFunctionVar::Multi(uf5, "uf5-2d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(6);
    test_functions_map.insert(
        "uf6",
        TestFunctionVar::Multi(uf6, "uf6-2d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(7);
    test_functions_map.insert(
        "uf7",
        TestFunctionVar::Multi(uf7, "uf7-2d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(8);
    test_functions_map.insert(
        "uf8",
        TestFunctionVar::Multi(uf8, "uf8-3d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(9);
    test_functions_map.insert(
        "uf9",
        TestFunctionVar::Multi(uf9, "uf9-3d", bounds.0, bounds.1),
    );
    let bounds = get_upper_bounds(10);
    test_functions_map.insert(
        "uf10",
        TestFunctionVar::Multi(uf10, "uf10-3d", bounds.0, bounds.1),
    );
}

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert(
        "uf",
        vec![
            "uf1".to_string(),
            "uf2".to_string(),
            "uf3".to_string(),
            "uf4".to_string(),
            "uf5".to_string(),
            "uf6".to_string(),
            "uf7".to_string(),
            "uf8".to_string(),
            "uf9".to_string(),
            "uf10".to_string(),
        ],
    );
}

fn get_upper_bounds(i: i8) -> (Vec<f64>, Vec<f64>) {
    let mut ub = vec![1.0];
    let mut lb = vec![0.0];
    match i {
        1 | 2 | 5 | 6 | 7 => {
            for _ in 1..30 {
                ub.push(1.0);
                lb.push(-1.0);
            }
        }
        3 => {
            for _ in 1..30 {
                ub.push(1.0);
                lb.push(0.0);
            }
        }
        4 => {
            for _ in 1..30 {
                ub.push(2.0);
                lb.push(-2.0);
            }
        }
        8 | 9 | 10 => {
            ub.push(1.0);
            lb.push(0.0);
            for _ in 2..30 {
                ub.push(2.0);
                lb.push(-2.0);
            }
        }
        _ => panic!("Test function does not exist"),
    }
    (lb, ub)
}

pub fn uf1(x: &Vec<f64>) -> Vec<f64> {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let odd = ((x.len() - 1) / 2) as f64;
    let even = (x.len() / 2) as f64;

    for i in 1..x.len() {
        let j = (i + 1) as f64;
        let a =
            (x[i] - (6.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64).sin()).powi(2);
        if (i + 1) % 2 == 1 {
            f1 += a;
        } else {
            f2 += a;
        }
    }

    f1 = x[0] + 2.0 / odd * f1;
    f2 = 1.0 - x[0].sqrt() + 2.0 / even * f2;
    vec![f1, f2]
}

pub fn uf2(x: &Vec<f64>) -> Vec<f64> {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let odd = ((x.len() - 1) / 2) as f64;
    let even = (x.len() / 2) as f64;

    for i in 1..x.len() {
        let j = (i + 1) as f64;
        let a = 0.3 * x[0].powi(2)
            * (24.0 * consts::PI * x[0] + (4.0 * j * consts::PI) / x.len() as f64).cos()
            + 0.6 * x[0];
        let b = 6.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64;
        if (i + 1) % 2 == 1 {
            f1 += (x[i] - a * b.cos()).powi(2);
        } else {
            f2 += (x[i] - a * b.sin()).powi(2);
        }
    }

    f1 = x[0] + 2.0 / odd * f1;
    f2 = 1.0 - x[0].sqrt() + 2.0 / even * f2;
    vec![f1, f2]
}

pub fn uf3(x: &Vec<f64>) -> Vec<f64> {
    let odd = ((x.len() - 1) / 2) as f64;
    let even = (x.len() / 2) as f64;
    let mut odd_sum = 0.0;
    let mut even_sum = 0.0;
    let mut odd_product = 1.0;
    let mut even_product = 1.0;

    for i in 1..x.len() {
        let j = (i + 1) as f64;
        let a = x[i] - x[0].powf(0.5 * (1.0 + (3.0 * (j - 2.0)) / (x.len() as f64 - 2.0)));
        let b = a.powi(2);
        let c = ((20.0 * a * consts::PI) / j.sqrt()).cos();
        if (i + 1) % 2 == 1 {
            odd_sum += b;
            odd_product *= c;
        } else {
            even_sum += b;
            even_product *= c;
        }
    }

    let f1 = x[0] + (2.0 / odd) * (4.0 * odd_sum - 2.0 * odd_product + 2.0);
    let f2 = 1.0 - x[0].sqrt() + (2.0 / even) * (4.0 * even_sum - 2.0 * even_product + 2.0);
    vec![f1, f2]
}

pub fn uf4(x: &Vec<f64>) -> Vec<f64> {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let odd = ((x.len() - 1) / 2) as f64;
    let even = (x.len() / 2) as f64;

    for i in 1..x.len() {
        let j = (i + 1) as f64;
        let a = x[i] - (6.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64).sin();
        let h = a.abs() / (1.0 + consts::E.powf(2.0 * a.abs()));
        if (i + 1) % 2 == 1 {
            f1 += h;
        } else {
            f2 += h;
        }
    }
    f1 = x[0] + 2.0 / odd * f1;
    f2 = 1.0 - x[0].powi(2) + 2.0 / even * f2;
    vec![f1, f2]
}

pub fn uf5(x: &Vec<f64>) -> Vec<f64> {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let n = 10.0;
    let epsilon = 0.1;
    let odd = ((x.len() - 1) / 2) as f64;
    let even = (x.len() / 2) as f64;

    for i in 1..x.len() {
        let j = (i + 1) as f64;
        let a = x[i] - (6.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64).sin();
        let h = 2.0 * a.powi(2) - (4.0 * consts::PI * a).cos() + 1.0;
        if (i + 1) % 2 == 1 {
            f1 += h;
        } else {
            f2 += h;
        }
    }
    f1 = x[0] + (1.0 / (2.0 * n) + epsilon) * (2.0 * n * consts::PI * x[0]).sin().abs()
        + 2.0 / odd * f1;
    f2 = 1.0 - x[0] + (1.0 / (2.0 * n) + epsilon) * (2.0 * n * consts::PI * x[0]).sin().abs()
        + 2.0 / even * f2;
    vec![f1, f2]
}

pub fn uf6(x: &Vec<f64>) -> Vec<f64> {
    let n = 2.0;
    let epsilon = 0.1;
    let odd = ((x.len() - 1) / 2) as f64;
    let even = (x.len() / 2) as f64;
    let mut odd_sum = 0.0;
    let mut even_sum = 0.0;
    let mut odd_product = 1.0;
    let mut even_product = 1.0;

    for i in 1..x.len() {
        let j = (i + 1) as f64;
        let a = x[i] - (6.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64).sin();
        let b = a.powi(2);
        let c = ((20.0 * a * consts::PI) / j.sqrt()).cos();
        if (i + 1) % 2 == 1 {
            odd_sum += b;
            odd_product *= c;
        } else {
            even_sum += b;
            even_product *= c;
        }
    }
    let f1 = x[0] + (2.0 * (1.0 / (2.0 * n) + epsilon) * (2.0 * n * consts::PI * x[0]).sin())
        .max(0.0) + 2.0 / odd * (4.0 * odd_sum - 2.0 * odd_product + 2.0);

    let f2 = 1.0 - x[0]
        + (2.0 * (1.0 / (2.0 * n) + epsilon) * (2.0 * n * consts::PI * x[0]).sin()).max(0.0)
        + 2.0 / even * (4.0 * even_sum - 2.0 * even_product + 2.0);
    vec![f1, f2]
}

pub fn uf7(x: &Vec<f64>) -> Vec<f64> {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let odd = ((x.len() - 1) / 2) as f64;
    let even = (x.len() / 2) as f64;

    for i in 1..x.len() {
        let j = (i + 1) as f64;
        let a = x[i] - (6.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64).sin();
        let b = a.powi(2);
        if (i + 1) % 2 == 1 {
            f1 += b;
        } else {
            f2 += b;
        }
    }
    f1 = x[0].powf(1.0 / 5.0) + 2.0 / odd * f1;
    f2 = 1.0 - x[0].powf(1.0 / 5.0) + 2.0 / even * f2;
    vec![f1, f2]
}

fn uf8_sums(x: &Vec<f64>) -> (f64, f64, f64) {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let mut f3 = 0.0;
    for i in 2..x.len() {
        let j = (i + 1) as f64;
        let a = (x[i] - 2.0 * x[1]
            * (2.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64).sin())
            .powi(2);
        if (i) % 3 == 0 {
            f1 += a;
        } else if (i + 1) % 3 == 0 {
            f3 += a;
        } else {
            f2 += a;
        }
    }
    (f1, f2, f3)
}

pub fn uf8(x: &Vec<f64>) -> Vec<f64> {
    let one = ((x.len() - 1) / 3) as f64;
    let two = ((x.len() - 2) / 3) as f64;
    let three = (x.len() / 3) as f64;
    let sums = uf8_sums(x);

    let f1 = (0.5 * x[0] * consts::PI).cos() * (0.5 * x[1] * consts::PI).cos() + 2.0 / one * sums.0;
    let f2 = (0.5 * x[0] * consts::PI).cos() * (0.5 * x[1] * consts::PI).sin() + 2.0 / two * sums.1;
    let f3 = (0.5 * x[0] * consts::PI).sin() + 2.0 / three * sums.2;
    vec![f1, f2, f3]
}

pub fn uf9(x: &Vec<f64>) -> Vec<f64> {
    let one = ((x.len() - 1) / 3) as f64;
    let two = ((x.len() - 2) / 3) as f64;
    let three = (x.len() / 3) as f64;
    let epsilon = 0.1;
    let sums = uf8_sums(x);

    let f1 = 0.5 * (((1.0 - epsilon) * (1.0 - 4.0 * (2.0 * x[0] - 1.0).powi(2))).max(0.0)
        + 2.0 * x[0]) * x[1] + 2.0 / one * sums.0;
    let f2 = 0.5 * (((1.0 - epsilon) * (1.0 - 4.0 * (2.0 * x[0] - 1.0).powi(2))).max(0.0)
        - 2.0 * x[0] + 2.0) * x[1] + 2.0 / two * sums.1;
    let f3 = 1.0 - x[1] + 2.0 / three * sums.2;
    vec![f1, f2, f3]
}

pub fn uf10(x: &Vec<f64>) -> Vec<f64> {
    let one = ((x.len() - 1) / 3) as f64;
    let two = ((x.len() - 2) / 3) as f64;
    let three = (x.len() / 3) as f64;
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let mut f3 = 0.0;
    for i in 2..x.len() {
        let j = (i + 1) as f64;
        let a =
            x[i] - 2.0 * x[1] * (2.0 * consts::PI * x[0] + (j * consts::PI) / x.len() as f64).sin();
        let b = 4.0 * a.powi(2) - (8.0 * consts::PI * a).cos() + 1.0;
        if (i) % 3 == 0 {
            f1 += b;
        } else if (i + 1) % 3 == 0 {
            f3 += b;
        } else {
            f2 += b;
        }
    }

    let f1 = (0.5 * x[0] * consts::PI).cos() * (0.5 * x[1] * consts::PI).cos() + 2.0 / one * f1;
    let f2 = (0.5 * x[0] * consts::PI).cos() * (0.5 * x[1] * consts::PI).sin() + 2.0 / two * f2;
    let f3 = (0.5 * x[0] * consts::PI).sin() + 2.0 / three * f3;
    vec![f1, f2, f3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{weak_rng, Rng};

    #[test]
    fn uf1_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let dimensions = d as f64;
            let x1 = rng.gen_range(0.0, 1.0);
            let mut input = vec![x1];
            for i in 1..dimensions as usize {
                input.push(
                    (6.0 * consts::PI * x1 + ((i + 1) as f64 * consts::PI) / dimensions).sin(),
                );
            }
            let result = uf1(&input);
            assert!(result[0] >= 0.0 && result[0] <= 1.0 && result[1] == 1.0 - result[0].sqrt());
        }
    }

    #[test]
    fn uf1_not_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let mut input = vec![rng.gen_range(-1.0, 1.0); d];
            input[0] = 1.5;
            let result = uf1(&input);
            assert!(result[0] < 0.0 || result[0] > 1.0 || result[1] != 1.0 - result[0].sqrt());
        }
    }

    #[test]
    fn uf2_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let dimensions = d as f64;
            let x1 = rng.gen_range(0.0, 1.0) as f64;
            let mut input = vec![x1];
            for i in 1..dimensions as usize {
                let a = 0.3 * x1.powi(2)
                    * (24.0 * consts::PI * x1 + (4.0 * (i + 1) as f64 * consts::PI) / dimensions)
                        .cos() + 0.6 * x1;
                let b = 6.0 * consts::PI * x1 + ((i + 1) as f64 * consts::PI) / dimensions;
                if (i + 1) % 2 == 1 {
                    input.push(a * b.cos());
                } else {
                    input.push(a * b.sin());
                }
            }
            let result = uf2(&input);
            assert!(result[0] >= 0.0 && result[0] <= 1.0 && result[1] == 1.0 - result[0].sqrt());
        }
    }

    #[test]
    fn uf2_not_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let mut input = vec![rng.gen_range(-1.0, 1.0); d];
            input[0] = 1.5;
            let result = uf2(&input);
            assert!(result[0] < 0.0 || result[0] > 1.0 || result[1] != 1.0 - result[0].sqrt());
        }
    }
    #[test]
    fn uf3_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let dimensions = d as f64;
            let x1 = rng.gen_range(0.0, 1.0) as f64;
            let mut input = vec![x1];
            for i in 1..dimensions as usize {
                input.push(x1.powf(
                    0.5 * (1.0 + (3.0 * ((i + 1) as f64 - 2.0)) / (dimensions - 2.0)),
                ));
            }
            let result = uf3(&input);
            assert!(result[0] >= 0.0 && result[0] <= 1.0 && result[1] == 1.0 - result[0].sqrt());
        }
    }

    #[test]
    fn uf3_not_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let mut input = vec![rng.gen_range(0.0, 1.0); d];
            input[0] = 1.5;
            let result = uf3(&input);
            assert!(result[0] < 0.0 || result[0] > 1.0 || result[1] != 1.0 - result[0].sqrt());
        }
    }

    #[test]
    fn uf4_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let dimensions = d as f64;
            let x1 = rng.gen_range(0.0, 1.0) as f64;
            let mut input = vec![x1];
            for i in 1..dimensions as usize {
                let j = (i + 1) as f64;
                input.push((6.0 * consts::PI * x1 + (j * consts::PI) / dimensions).sin());
            }
            let result = uf4(&input);
            assert!(result[0] >= 0.0 && result[0] <= 1.0 && result[1] == 1.0 - result[0].powi(2));
        }
    }
    #[test]
    fn uf4_not_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let mut input = vec![rng.gen_range(-2.0, 2.0); d];
            input[0] = 1.5;
            let result = uf4(&input);
            assert!(result[0] < 0.0 || result[0] > 1.0 || result[1] != 1.0 - result[0].powi(2));
        }
    }

    #[test]
    fn uf7_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let dimensions = d as f64;
            let x1 = rng.gen_range(0.0, 1.0) as f64;
            let mut input = vec![x1];
            for i in 1..dimensions as usize {
                let j = (i + 1) as f64;
                input.push((6.0 * consts::PI * x1 + (j * consts::PI) / dimensions).sin());
            }
            let result = uf7(&input);
            assert!(result[0] >= 0.0 && result[0] <= 1.0 && result[1] == 1.0 - result[0]);
        }
    }

    #[test]
    fn uf7_not_optimum() {
        let mut rng = weak_rng();
        for d in 3..31 {
            let mut input = vec![rng.gen_range(-1.0, 1.0); d];
            input[0] = 1.5;
            let result = uf7(&input);
            assert!(result[0] < 0.0 || result[0] > 1.0 || result[1] != 1.0 - result[0]);
        }
    }
    #[test]
    fn uf8_optimum() {
        let mut rng = weak_rng();
        for d in 5..31 {
            let dimensions = d as f64;
            let x1 = rng.gen_range(0.0, 1.0) as f64;
            let x2 = rng.gen_range(0.0, 1.0) as f64;
            let mut input = vec![x1, x2];
            for i in 2..dimensions as usize {
                let j = (i + 1) as f64;
                input
                    .push(2.0 * x2 * (2.0 * consts::PI * x1 + (j * consts::PI) / dimensions).sin());
            }
            let result = uf8(&input);
            assert!(result[0] >= 0.0 && result[0] <= 1.0);
            assert!(result[1] >= 0.0 && result[1] <= 1.0);
            assert!(result[2] >= 0.0 && result[2] <= 1.0);
            assert_approx_eq!(
                result[0].powi(2) + result[1].powi(2) + result[2].powi(2),
                1.0
            );
        }
    }

    #[test]
    fn uf8_not_optimum() {
        let mut rng = weak_rng();
        for d in 5..31 {
            let mut input = vec![rng.gen_range(-2.0, 2.0); d];
            input[0] = 1.5;
            input[1] = 1.5;
            let result = uf8(&input);
            assert!(
                result[0].powi(2) + result[1].powi(2) + result[2].powi(2) != 1.0 || result[0] < 0.0
                    || result[0] > 1.0 || result[1] < 0.0 || result[1] > 1.0
                    || result[2] < 0.0 || result[2] > 1.0
            );
        }
    }

    #[test]
    fn uf9_optimum() {
        let mut rng = weak_rng();
        for d in 5..31 {
            let dimensions = d as f64;
            let x1 = if rng.next_f64() > 0.5 {
                rng.gen_range(0.0, 0.25) as f64
            } else {
                rng.gen_range(0.75, 1.0) as f64
            };
            let x2 = rng.gen_range(0.0, 1.0) as f64;
            let mut input = vec![x1, x2];
            for i in 2..dimensions as usize {
                let j = (i + 1) as f64;
                input
                    .push(2.0 * x2 * (2.0 * consts::PI * x1 + (j * consts::PI) / dimensions).sin());
            }
            let result = uf9(&input);
            assert!(
                (result[0] >= 0.0 && result[0] <= 1.0 / 4.0 * (1.0 - result[2]))
                    || (3.0 / 4.0 * (1.0 - result[2]) <= result[0] && result[0] <= 1.0)
            );
            assert_approx_eq!(result[1], 1.0 - result[0] - result[2]);

            assert!(result[2] >= 0.0 && result[2] <= 1.0);
        }
    }

    #[test]
    fn uf10_optimum() {
        let mut rng = weak_rng();
        for d in 5..31 {
            let dimensions = d as f64;
            let x1 = if rng.next_f64() > 0.5 {
                rng.gen_range(0.0, 0.25) as f64
            } else {
                rng.gen_range(0.75, 1.0) as f64
            };
            let x2 = rng.gen_range(0.0, 1.0) as f64;
            let mut input = vec![x1, x2];
            for i in 2..dimensions as usize {
                let j = (i + 1) as f64;
                input
                    .push(2.0 * x2 * (2.0 * consts::PI * x1 + (j * consts::PI) / dimensions).sin());
            }
            let result = uf10(&input);
            assert!(result[0] >= 0.0 && result[0] <= 1.0);
            assert!(result[1] >= 0.0 && result[1] <= 1.0);
            assert!(result[2] >= 0.0 && result[2] <= 1.0);
            assert_approx_eq!(
                result[0].powi(2) + result[1].powi(2) + result[2].powi(2),
                1.0
            );
        }
    }
}
