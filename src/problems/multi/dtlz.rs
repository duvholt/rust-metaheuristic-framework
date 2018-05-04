use fitness_evaluation::TestFunctionVar;
use std::collections::HashMap;
use std::f64::consts;

pub fn add_test_functions(test_functions_map: &mut HashMap<&'static str, TestFunctionVar>) {
    test_functions_map.insert(
        "dtlz1",
        TestFunctionVar::Multi(dtlz1, "dtlz1-3d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "dtlz2",
        TestFunctionVar::Multi(dtlz2, "dtlz2-3d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "dtlz3",
        TestFunctionVar::Multi(dtlz3, "dtlz3-3d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "dtlz4",
        TestFunctionVar::Multi(dtlz4, "dtlz4-3d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "dtlz5",
        TestFunctionVar::Multi(dtlz5, "dtlz5-3d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "dtlz6",
        TestFunctionVar::Multi(dtlz6, "dtlz6-3d", vec![0.0; 30], vec![1.0; 30]),
    );
    test_functions_map.insert(
        "dtlz7",
        TestFunctionVar::Multi(dtlz7, "dtlz7-3d", vec![0.0; 30], vec![1.0; 30]),
    );
}

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert(
        "dtlz",
        vec![
            "dtlz1".to_string(),
            "dtlz2".to_string(),
            "dtlz3".to_string(),
            "dtlz4".to_string(),
            "dtlz5".to_string(),
            "dtlz6".to_string(),
            "dtlz7".to_string(),
        ],
    );
}

fn dtlz1g(x: &Vec<f64>, m: usize) -> f64 {
    let k = x.len() - m + 1;
    100.0
        * (k as f64
            + (x.len() - k..x.len())
                .map(|i| (x[i] - 0.5).powi(2) - (20.0 * consts::PI * (x[i] - 0.5)).cos())
                .sum::<f64>())
}

fn dtlz2g(x: &Vec<f64>, m: usize) -> f64 {
    let k = x.len() - m + 1;
    (x.len() - k..x.len()).map(|i| (x[i] - 0.5).powi(2)).sum()
}

fn dtlz6g(x: &Vec<f64>, m: usize) -> f64 {
    let k = x.len() - m + 1;
    (x.len() - k..x.len()).map(|i| x[i].powf(0.1)).sum()
}

fn dtlz7g(x: &Vec<f64>, m: usize) -> f64 {
    let k = x.len() - m + 1;
    1.0 + 9.0 / k as f64 * (x.len() - k..x.len()).map(|i| x[i]).sum::<f64>()
}

fn dtlz4y(x: &Vec<f64>) -> Vec<f64> {
    x.iter().map(|i| i.powi(2)).collect()
}

fn dtlz5y(x: &Vec<f64>, m: usize, g: f64) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if i > 0 && i < m - 1 {
                (1.0 + 2.0 * g * x[i]) / (2.0 * (1.0 + g))
            } else {
                x[i]
            }
        })
        .collect()
}

fn dtlz2_6(x: &Vec<f64>, v: i8) -> Vec<f64> {
    let mut result = vec![];
    let m = 3;
    let g = match v {
        2 | 4 | 5 => dtlz2g(x, m),
        3 => dtlz1g(x, m),
        6 => dtlz6g(x, m),
        _ => panic!("Test function does not exist"),
    };
    let y = match v {
        2 | 3 => x.clone(),
        4 => dtlz4y(x),
        5 | 6 => dtlz5y(x, m, g),
        _ => panic!("Test function does not exist"),
    };
    //f_1
    result.push(
        (1.0 + g)
            * (0..m - 1)
                .map(|i| (y[i] * consts::PI / 2.0).cos())
                .product::<f64>(),
    );
    //f_2 to f_m-1
    for i in 1..m - 1 {
        let product: f64 = (0..m - (i + 1))
            .map(|j| (y[j] * consts::PI / 2.0).cos())
            .product();
        result.push((1.0 + g) * product * (y[m - (i + 1)] * consts::PI / 2.0).sin());
    }
    //f_m
    result.push((1.0 + g) * (y[0] * consts::PI / 2.0).sin());
    result
}

pub fn dtlz1(x: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![];
    let m = 3;
    let g = dtlz1g(x, m);
    //f_1
    result.push((1.0 + g) * 0.5 * (0..m - 1).map(|i| x[i]).product::<f64>());
    //f_2 to f_m-1
    for i in 1..m - 1 {
        let product: f64 = (0..m - (i + 1)).map(|j| x[j]).product();
        result.push((1.0 + g) * 0.5 * product * (1.0 - x[m - (i + 1)]));
    }
    //f_m
    result.push((1.0 + g) * 0.5 * (1.0 - x[0]));
    result
}

pub fn dtlz2(x: &Vec<f64>) -> Vec<f64> {
    dtlz2_6(x, 2)
}

pub fn dtlz3(x: &Vec<f64>) -> Vec<f64> {
    dtlz2_6(x, 3)
}

pub fn dtlz4(x: &Vec<f64>) -> Vec<f64> {
    dtlz2_6(x, 4)
}

pub fn dtlz5(x: &Vec<f64>) -> Vec<f64> {
    dtlz2_6(x, 5)
}

pub fn dtlz6(x: &Vec<f64>) -> Vec<f64> {
    dtlz2_6(x, 6)
}

pub fn dtlz7(x: &Vec<f64>) -> Vec<f64> {
    let m = 3;
    let g = dtlz7g(x, m);
    //f_1 to f_m-1
    let mut result = x[..m - 1].to_vec();
    //f_m
    let sum = (0..m - 1)
        .map(|i| result[i] / (1.0 + g) * (1.0 + (3.0 * consts::PI * result[i]).sin()))
        .sum::<f64>();
    result.push((1.0 + g) * (m as f64 - sum));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{weak_rng, Rng};

    fn sum(vector: Vec<f64>) -> f64 {
        vector.iter().sum::<f64>()
    }

    fn sum_pow(vector: Vec<f64>) -> f64 {
        vector.iter().map(|i| i.powi(2)).sum::<f64>()
    }

    fn dtlz_optimum(dtlz: &Fn(&Vec<f64>) -> Vec<f64>, sum: &Fn(Vec<f64>) -> f64, x: f64, f: f64) {
        let mut rng = weak_rng();
        let objectives = 3;
        for i in objectives..31 {
            let mut vector = vec![x; i];
            let k = i - objectives + 1;
            for j in 0..i - k {
                vector[j] = rng.next_f64();
                let result = dtlz(&vector);
                assert_approx_eq!(sum(result), f);
            }
        }
    }

    fn dtlz_not_optimum(dtlz: &Fn(&Vec<f64>) -> Vec<f64>, sum: &Fn(Vec<f64>) -> f64, f: f64) {
        let result = dtlz(&vec![0.5, 0.5, 0.55, 0.5]);
        assert_ne!(sum(result), f);
        let result = dtlz(&vec![0.5, 0.5, 0.5, 0.8]);
        assert_ne!(sum(result), f);
        let result = dtlz(&vec![0.5, 0.5, 0.55, 0.1]);
        assert_ne!(sum(result), f);
        let result = dtlz(&vec![0.5, 0.5, 0.6, 0.6]);
        assert_ne!(sum(result), f);
    }
    #[test]
    fn dtlz1_optimum() {
        dtlz_optimum(&dtlz1, &sum, 0.5, 0.5);
    }

    #[test]
    fn dtlz1_not_optimum() {
        dtlz_not_optimum(&dtlz1, &sum, 0.5);
    }

    #[test]
    fn dtlz2_optimum() {
        dtlz_optimum(&dtlz2, &sum_pow, 0.5, 1.0);
    }

    #[test]
    fn dtlz2_not_optimum() {
        dtlz_not_optimum(&dtlz2, &sum_pow, 1.0);
    }

    #[test]
    fn dtlz3_optimum() {
        dtlz_optimum(&dtlz3, &sum_pow, 0.5, 1.0);
    }

    #[test]
    fn dtlz3_not_optimum() {
        dtlz_not_optimum(&dtlz3, &sum_pow, 1.0);
    }

    #[test]
    fn dtlz4_optimum() {
        dtlz_optimum(&dtlz4, &sum_pow, 0.5, 1.0);
    }

    #[test]
    fn dtlz4_not_optimum() {
        dtlz_not_optimum(&dtlz4, &sum_pow, 1.0);
    }

    #[test]
    fn dtlz5_optimum() {
        dtlz_optimum(&dtlz5, &sum_pow, 0.5, 1.0);
    }

    #[test]
    fn dtlz5_not_optimum() {
        dtlz_not_optimum(&dtlz5, &sum_pow, 1.0);
    }

    #[test]
    fn dtlz6_optimum() {
        dtlz_optimum(&dtlz6, &sum_pow, 0.0, 1.0);
    }

    #[test]
    fn dtlz6_not_optimum() {
        dtlz_not_optimum(&dtlz6, &sum_pow, 1.0);
    }

}
