use fitness_evaluation::TestFunctionVar;
use std::collections::HashMap;
use std::f64::consts;

pub fn add_test_functions(test_functions_map: &mut HashMap<&'static str, TestFunctionVar>) {
    test_functions_map.insert("zakharov", TestFunctionVar::Single(zakharov));
    test_functions_map.insert("himmelblau", TestFunctionVar::Single(himmelblau));
    test_functions_map.insert("sphere", TestFunctionVar::Single(sphere));
    test_functions_map.insert(
        "hyper-ellipsoid",
        TestFunctionVar::Single(axis_parallel_hyper_ellipsoid),
    );
    test_functions_map.insert(
        "moved-hyper-ellipsoid",
        TestFunctionVar::Single(moved_axis_parallel_hyper_ellipsoid),
    );
    test_functions_map.insert("levy05", TestFunctionVar::Single(levy05));
    test_functions_map.insert("easom", TestFunctionVar::Single(easom));
}

pub fn zakharov(x: &Vec<f64>) -> f64 {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;

    let mut i = 1.0;
    x.iter().for_each(|xi| {
        sum1 = sum1 + xi.powf(2.0);
        sum2 = sum2 + 0.5 * i * xi;
        i += 1.0;
    });
    return sum1 + sum2.powf(2.0) + sum2.powf(4.0);
}

pub fn himmelblau(x: &Vec<f64>) -> f64 {
    if x.len() != 2 {
        panic!("Himmelblau only supports two dimensions!");
    }
    let x_1 = x[0];
    let x_2 = x[1];
    return (x_1.powf(2.0) + x_2 - 11.0).powf(2.0) + (x_1 + x_2.powf(2.0) - 7.0).powf(2.0);
}

pub fn axis_parallel_hyper_ellipsoid(x: &Vec<f64>) -> f64 {
    (0..x.len()).map(|i| (i as f64 + 1.0) * x[i].powi(2)).sum()
}

pub fn moved_axis_parallel_hyper_ellipsoid(x: &Vec<f64>) -> f64 {
    (0..x.len())
        .map(|i| {
            let a = i as f64 + 1.0;
            (a * (x[i] - 5.0 * a)).powi(2)
        })
        .sum()
}

pub fn sphere(x: &Vec<f64>) -> f64 {
    (0..x.len()).map(|i| x[i].powi(2)).sum()
}

pub fn levy05(x: &Vec<f64>) -> f64 {
    if x.len() != 2 {
        panic!("Levy05 only supports two dimensions!");
    }
    let sum_0 = (1..6)
        .map(|i| i as f64)
        .map(|i| i * ((i - 1.0) * x[0] + i).cos())
        .sum::<f64>();
    let sum_1 = (1..6)
        .map(|i| i as f64)
        .map(|i| i * ((i + 1.0) * x[1] + i).cos())
        .sum::<f64>();
    return sum_0 * sum_1 + (x[0] + 1.42513).powi(2) + (x[1] + 0.080032).powi(2);
}

pub fn easom(x: &Vec<f64>) -> f64 {
    let cos_product: f64 = x.iter().map(|x_i| x_i.cos()).product();
    let sum: f64 = x.iter().map(|x_i| -(x_i - consts::PI).powi(2)).sum();
    -cos_product * sum.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    #[test]
    fn sphere_optimum() {
        assert_eq!(sphere(&vec![0.0]), 0.0);
        assert_eq!(sphere(&vec![0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn sphere_not_optimum() {
        assert_ne!(sphere(&vec![2.0]), 0.0);
        assert_ne!(sphere(&vec![0.1, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn axis_parallel_hyper_ellipsoid_optimum() {
        assert_eq!(
            axis_parallel_hyper_ellipsoid(&vec![0.0, 0.0, 0.0, 0.0]),
            0.0
        );
        assert_eq!(axis_parallel_hyper_ellipsoid(&vec![0.0, 0.0]), 0.0);
        assert_eq!(axis_parallel_hyper_ellipsoid(&vec![0.0]), 0.0);
    }

    #[test]
    fn axis_parallel_hyper_ellipsoid_not_optimum() {
        assert_ne!(axis_parallel_hyper_ellipsoid(&vec![1.0, 0.0, 0.0]), 0.0);
        assert_ne!(axis_parallel_hyper_ellipsoid(&vec![0.1]), 0.0);
    }

    #[test]
    fn moved_axis_parallel_hyper_ellipsoid_optimum() {
        assert_eq!(
            moved_axis_parallel_hyper_ellipsoid(&vec![5.0, 10.0, 15.0, 20.0]),
            0.0
        );
        assert_eq!(moved_axis_parallel_hyper_ellipsoid(&vec![5.0]), 0.0);
        assert_eq!(moved_axis_parallel_hyper_ellipsoid(&vec![5.0, 10.0]), 0.0);
    }

    #[test]
    fn moved_axis_parallel_hyper_ellipsoid_not_optimum() {
        assert_ne!(
            moved_axis_parallel_hyper_ellipsoid(&vec![5.0, 10.0, 5.0]),
            0.0
        );
        assert_ne!(moved_axis_parallel_hyper_ellipsoid(&vec![5.1]), 0.0);
    }

    #[test]
    fn zakharov_optimum() {
        assert_eq!(0.0, zakharov(&vec![0.0, 0.0]));
        assert_eq!(0.0, zakharov(&vec![0.0, 0.0, 0.0]));
        assert_eq!(0.0, zakharov(&vec![0.0, 0.0, 0.0, 0.0]));
        assert_eq!(0.0, zakharov(&vec![0.0, 0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn zakharov_not_optimum() {
        assert_ne!(0.0, zakharov(&vec![2.0, -1.3]));
    }

    #[test]
    fn himmelblau_optimum() {
        assert_eq!(0.0, himmelblau(&vec![3.0, 2.0]));
        assert_approx_eq!(0.0, himmelblau(&vec![-2.805118, 3.131312]));
        assert_approx_eq!(0.0, himmelblau(&vec![-3.779310, -3.283186]));
        assert_approx_eq!(0.0, himmelblau(&vec![3.584428, -1.848126]));
    }

    #[test]
    fn himmelblau_not_optimum() {
        assert_ne!(0.0, himmelblau(&vec![4.0, 6.0]));
    }

    #[test]
    fn levy05_optimum() {
        assert_approx_eq!(-174.71914553453792, levy05(&vec![-1.3068, -1.4248]));
    }

    #[test]
    fn levy05_not_optimum() {
        assert_ne!(0.0, levy05(&vec![-1.0, -1.0]));
    }

    #[test]
    fn easom_optimum() {
        assert_approx_eq!(-1.0, easom(&vec![consts::PI, consts::PI]));
    }

    #[test]
    fn easom_not_optimum() {
        assert_approx_eq!(0.0, easom(&vec![-1.0, -1.0]));
    }

    #[ignore]
    #[bench]
    fn bench_zakharov(b: &mut Bencher) {
        let x = vec![1.0, 3.0];
        b.iter(|| {
            zakharov(&x);
        });
    }

    #[ignore]
    #[bench]
    fn bench_himmelblau(b: &mut Bencher) {
        let x = vec![1.0, 3.0];
        b.iter(|| {
            himmelblau(&x);
        });
    }
}
