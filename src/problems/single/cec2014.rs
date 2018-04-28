use config::AlgorithmInfo;
use fitness_evaluation::TestFunctionVar;
use std::collections::HashMap;
use std::f64::consts;

pub fn add_test_functions(test_functions_map: &mut HashMap<&'static str, TestFunctionVar>) {
    test_functions_map.insert("rosenbrock", TestFunctionVar::Single(rosenbrock));
    test_functions_map.insert("high-elliptic", TestFunctionVar::Single(high_elliptic));
    test_functions_map.insert("bent-cigar", TestFunctionVar::Single(bent_cigar));
    test_functions_map.insert("griewank", TestFunctionVar::Single(griewank));
    test_functions_map.insert("schwefel", TestFunctionVar::Single(schwefel));
    test_functions_map.insert("ackley", TestFunctionVar::Single(ackley));
    test_functions_map.insert("rastrigin", TestFunctionVar::Single(rastrigin));
    test_functions_map.insert("katsuura", TestFunctionVar::Single(katsuura));
    test_functions_map.insert("weierstrass", TestFunctionVar::Single(weierstrass));
    test_functions_map.insert("happycat", TestFunctionVar::Single(happycat));
    test_functions_map.insert("hgbat", TestFunctionVar::Single(hgbat));
    test_functions_map.insert("discus", TestFunctionVar::Single(discus));
    test_functions_map.insert(
        "griewank-rosenbrock",
        TestFunctionVar::Single(griewank_rosenbrock),
    );
    test_functions_map.insert(
        "expanded-schaffer6",
        TestFunctionVar::Single(expanded_schaffer6),
    );
}

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert(
        "cec2014",
        vec![
            "high-elliptic".to_string(),
            "bent-cigar".to_string(),
            "discus".to_string(),
            "rosenbrock".to_string(),
            "ackley".to_string(),
            "weierstrass".to_string(),
            "griewank".to_string(),
            "rastrigin".to_string(),
            "schwefel".to_string(),
            "katsuura".to_string(),
            "happycat".to_string(),
            "hgbat".to_string(),
            "griewank-rosenbrock".to_string(),
            "expanded-schaffer6".to_string(),
        ],
    );
}

pub fn algorithm_shift_info(name: &str) -> Option<AlgorithmInfo> {
    match name {
        "high-elliptic" => Some(AlgorithmInfo {
            number: 1,
            scale: 1.0,
            add: 0.0,
        }),
        "bent-cigar" => Some(AlgorithmInfo {
            number: 2,
            scale: 1.0,
            add: 0.0,
        }),
        "discus" => Some(AlgorithmInfo {
            number: 3,
            scale: 1.0,
            add: 0.0,
        }),
        "rosenbrock" => Some(AlgorithmInfo {
            number: 4,
            scale: 2.048 / 100.0,
            add: 1.0,
        }),
        "ackley" => Some(AlgorithmInfo {
            number: 5,
            scale: 1.0,
            add: 0.0,
        }),
        "weierstrass" => Some(AlgorithmInfo {
            number: 6,
            scale: 0.5 / 100.0,
            add: 0.0,
        }),
        "griewank" => Some(AlgorithmInfo {
            number: 7,
            scale: 600.0 / 100.0,
            add: 0.0,
        }),
        // "rastrigin" => Some(AlgorithmInfo{
        //     number: 8, scale: 5.12 / 100.0, add: 0.0
        // }),
        "rastrigin" => Some(AlgorithmInfo {
            number: 9,
            scale: 5.12 / 100.0,
            add: 0.0,
        }),
        // "schwefel" => Some(AlgorithmInfo{
        //     number: 10, scale: 1000.0 / 100.0, add: 0.0
        // }),
        "schwefel" => Some(AlgorithmInfo {
            number: 11,
            scale: 1000.0 / 100.0,
            add: 0.0,
        }),
        "katsuura" => Some(AlgorithmInfo {
            number: 12,
            scale: 5.0 / 100.0,
            add: 0.0,
        }),
        "happycat" => Some(AlgorithmInfo {
            number: 13,
            scale: 5.0 / 100.0,
            add: -1.0,
        }),
        "hgbat" => Some(AlgorithmInfo {
            number: 14,
            scale: 5.0 / 100.0,
            add: -1.0,
        }),
        "griewank-rosenbrock" => Some(AlgorithmInfo {
            number: 15,
            scale: 5.0 / 100.0,
            add: 1.0,
        }),
        "expanded-schaffer6" => Some(AlgorithmInfo {
            number: 16,
            scale: 1.0,
            add: 0.0,
        }),
        _ => None,
    }
}

pub fn ackley(x: &Vec<f64>) -> f64 {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * consts::PI;
    let d = x.len() as f64;
    let pow_sum: f64 = x.iter().map(|x_i| x_i.powf(2.0)).sum();
    let cos_sum: f64 = x.iter().map(|x_i| (c * x_i).cos()).sum();
    return -a * (-b * ((1.0 / d) * (pow_sum)).sqrt()).exp() - ((1.0 / d) * (cos_sum)).exp() + a
        + consts::E;
}

pub fn rosenbrock(x: &Vec<f64>) -> f64 {
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}

pub fn rastrigin(x: &Vec<f64>) -> f64 {
    let a = 10.0;
    let sum: f64 = (0..x.len())
        .map(|i| x[i].powi(2) - a * (2.0 * consts::PI * x[i]).cos())
        .sum();
    a * x.len() as f64 + sum
}

// High Conditioned Elliptic
pub fn high_elliptic(x: &Vec<f64>) -> f64 {
    let d = x.len();
    x.iter()
        .enumerate()
        .map(|(i, x_i)| (10.0f64.powi(6)).powf(i as f64 / (d - 1) as f64) * x_i.powi(2))
        .sum()
}

pub fn bent_cigar(x: &Vec<f64>) -> f64 {
    x[0].powi(2) + 10f64.powi(6) * x.iter().skip(1).map(|x_i| x_i.powi(2)).sum::<f64>()
}

pub fn griewank(x: &Vec<f64>) -> f64 {
    (1.0 / 4000.0) * x.iter().map(|x_i| x_i.powi(2)).sum::<f64>()
        - x.iter()
            .enumerate()
            .map(|(i, x_i)| (x_i / (i as f64 + 1.0).sqrt()).cos())
            .product::<f64>() + 1.0
}

pub fn schwefel(x: &Vec<f64>) -> f64 {
    let z = 4.209687462275036e+002;
    let d = x.len() as f64;
    418.9828872724338 * d
        - x.iter()
            .map(|x_i| {
                let z_i = x_i + z;
                if z_i < -500.0 {
                    (z_i.abs() % 500.0 - 500.0) * ((z_i.abs() % 500.0 - 500.0).abs()).sqrt().sin()
                        - (z_i + 500.0).powi(2) / (10_000.0 * d)
                } else if z_i > 500.0 {
                    (500.0 - (z_i % 500.0)) * (500.0 - z_i % 500.0).abs().sqrt().sin()
                        - (z_i - 500.0).powi(2) / (10_000.0 * d)
                } else {
                    z_i * (z_i.abs().powf(1.0 / 2.0)).sin()
                }
            })
            .sum::<f64>()
}

pub fn katsuura(x: &Vec<f64>) -> f64 {
    let d = x.len() as f64;
    (10.0 / d / d)
        * x.iter()
            .enumerate()
            .map(|(i, x_i)| {
                (1.0 + (i as f64 + 1.0)
                    * (1..33)
                        .map(|j| {
                            let p = 2f64.powi(j) * x_i;
                            (p - (p + 0.5).floor()).abs() / 2f64.powi(j)
                        })
                        .sum::<f64>())
                    .powf(10.0 / d.powf(1.2))
            })
            .product::<f64>() - 10.0 / d / d
}

pub fn weierstrass(x: &Vec<f64>) -> f64 {
    let a = 0.5f64;
    let b = 3.0f64;
    let kmax = 20;
    let d = x.len() as f64;
    x.iter()
        .map(|x_i| {
            (0..kmax + 1)
                .map(|k| a.powi(k) * (2.0 * consts::PI * b.powi(k) * (x_i + 0.5)).cos())
                .sum::<f64>()
        })
        .sum::<f64>()
        - d * (0..kmax + 1)
            .map(|k| a.powi(k) * (2.0 * consts::PI * b.powi(k) * 0.5).cos())
            .sum::<f64>()
}

pub fn happycat(x: &Vec<f64>) -> f64 {
    let d = x.len() as f64;
    let pow_sum = x.iter().map(|x_i| x_i.powi(2)).sum::<f64>();
    (pow_sum - d).powi(2).powf(1.0 / 8.0) + (0.5 * pow_sum + x.iter().sum::<f64>()) / d + 0.5
}

pub fn hgbat(x: &Vec<f64>) -> f64 {
    let d = x.len() as f64;
    let pow_sum = x.iter().map(|x_i| x_i.powi(2)).sum::<f64>();
    let sum = x.iter().sum::<f64>();
    (pow_sum.powi(2) - sum.powi(2)).powi(2).powf(1.0 / 4.0) + (0.5 * pow_sum + sum) / d + 0.5
}

pub fn discus(x: &Vec<f64>) -> f64 {
    10.0_f64.powi(6) * x[0].powi(2) + (1..x.len()).map(|i| x[i].powi(2)).sum::<f64>()
}

pub fn schaffer6(x: &Vec<f64>) -> f64 {
    if x.len() != 2 {
        panic!("Schaffer6 only supports two dimensions!");
    }
    0.5 + ((x[0].powi(2) + x[1].powi(2)).sqrt().sin().powi(2) - 0.5)
        / (1.0 + 0.001 * x[0].powi(2) + 0.001 * x[1].powi(2)).powi(2)
}

pub fn expanded_schaffer6(x: &Vec<f64>) -> f64 {
    (0..x.len())
        .map(|i| schaffer6(&vec![x[i], x[(i + 1) % x.len()]]))
        .sum()
}

pub fn griewank_rosenbrock(x: &Vec<f64>) -> f64 {
    (0..x.len())
        .map(|i| {
            let x_current = x[i];
            let x_next = x[(i + 1) % x.len()];
            griewank(&vec![rosenbrock(&vec![x_current, x_next])])
        })
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn expanded_schaffer6_optimum() {
        assert_eq!(expanded_schaffer6(&vec![0.0, 0.0, 0.0]), 0.0);
        assert_eq!(expanded_schaffer6(&vec![0.0, 0.0]), 0.0);
    }

    #[test]
    fn expanded_schaffer6_not_optimum() {
        assert_ne!(expanded_schaffer6(&vec![0.1, 0.2, 0.0]), 0.0);
        assert_ne!(expanded_schaffer6(&vec![0.1, 0.0]), 0.0);
    }

    #[test]
    fn discus_optimum() {
        assert_eq!(discus(&vec![0.0, 0.0, 0.0]), 0.0);
        assert_eq!(discus(&vec![0.0, 0.0]), 0.0);
    }

    #[test]
    fn discus_not_optimum() {
        assert_ne!(discus(&vec![0.1, 0.0, 0.0]), 0.0);
    }
    #[test]
    fn rastrigin_optimum() {
        assert_eq!(rastrigin(&vec![0.0]), 0.0);
        assert_eq!(rastrigin(&vec![0.0, 0.0]), 0.0);
    }

    #[test]
    fn rastrigin_not_optimum() {
        assert_ne!(rastrigin(&vec![1.0]), 0.0);
        assert_ne!(rastrigin(&vec![0.0, 0.1]), 0.0);
    }

    #[test]
    fn rosenbrock_optimum() {
        assert_eq!(0.0, rosenbrock(&vec![1.0, 1.0]));
        assert_eq!(0.0, rosenbrock(&vec![1.0, 1.0, 1.0]));
        assert_eq!(0.0, rosenbrock(&vec![1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn rosenbrock_not_optimum() {
        assert_ne!(0.0, rosenbrock(&vec![3.0, 2.0]));
    }

    #[test]
    fn ackley_optimum() {
        assert_approx_eq!(0.0, ackley(&vec![0.0, 0.0]));
        assert_approx_eq!(0.0, ackley(&vec![0.0, 0.0, 0.0]));
        assert_approx_eq!(0.0, ackley(&vec![0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn ackley_not_optimum() {
        assert_ne!(0.0, ackley(&vec![2.0, -1.3]));
    }

    #[test]
    fn high_elliptic_optimum() {
        assert_approx_eq!(0.0, high_elliptic(&vec![0.0, 0.0]));
        assert_approx_eq!(0.0, high_elliptic(&vec![0.0, 0.0, 0.0]));
        assert_approx_eq!(0.0, high_elliptic(&vec![0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn high_elliptic_not_optimum() {
        assert_ne!(0.0, high_elliptic(&vec![2.0, -1.3]));
    }

    #[test]
    fn bent_cigar_optimum() {
        assert_approx_eq!(0.0, bent_cigar(&vec![0.0, 0.0]));
        assert_approx_eq!(0.0, bent_cigar(&vec![0.0, 0.0, 0.0]));
        assert_approx_eq!(0.0, bent_cigar(&vec![0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn bent_cigar_not_optimum() {
        assert_ne!(0.0, bent_cigar(&vec![2.0, -1.3]));
    }

    #[test]
    fn griewank_optimum() {
        assert_approx_eq!(0.0, griewank(&vec![0.0, 0.0]));
        assert_approx_eq!(0.0, griewank(&vec![0.0, 0.0, 0.0]));
        assert_approx_eq!(0.0, griewank(&vec![0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn griewank_not_optimum() {
        assert_ne!(0.0, griewank(&vec![2.0, -1.3]));
    }

    #[test]
    fn schwefel_optimum() {
        assert_approx_eq!(0.0, schwefel(&vec![0.0, 0.0]));
        assert_approx_eq!(0.0, schwefel(&vec![0.0, 0.0, 0.0]));
        assert_approx_eq!(0.0, schwefel(&vec![0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn schwefel_not_optimum() {
        assert_ne!(0.0, schwefel(&vec![2.0, -1.3]));
    }

    #[test]
    fn katsuura_optimum() {
        assert_approx_eq!(0.0, katsuura(&vec![0.0, 0.0]));
        assert_approx_eq!(0.0, katsuura(&vec![0.0, 0.0, 0.0]));
        assert_approx_eq!(0.0, katsuura(&vec![0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn katsuura_not_optimum() {
        assert_ne!(0.0, katsuura(&vec![2.0, -1.3, 5.0]));
    }

    #[test]
    fn weierstrass_optimum() {
        assert_approx_eq!(0.0, weierstrass(&vec![0.0, 0.0]));
        assert_approx_eq!(0.0, weierstrass(&vec![0.0, 0.0, 0.0]));
        assert_approx_eq!(0.0, weierstrass(&vec![0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn weierstrass_not_optimum() {
        assert_ne!(0.0, weierstrass(&vec![2.0, -1.3, 5.0]));
    }

    #[test]
    fn happycat_optimum() {
        assert_approx_eq!(0.0, happycat(&vec![-1.0, -1.0]));
        assert_approx_eq!(0.0, happycat(&vec![-1.0, -1.0, -1.0]));
        assert_approx_eq!(0.0, happycat(&vec![-1.0, -1.0, -1.0, -1.0]));
    }

    #[test]
    fn happycat_not_optimum() {
        assert_ne!(0.0, happycat(&vec![2.0, -1.3, 5.0]));
    }

    #[test]
    fn hgbat_optimum() {
        assert_approx_eq!(0.0, hgbat(&vec![-1.0, -1.0]));
        assert_approx_eq!(0.0, hgbat(&vec![-1.0, -1.0, -1.0]));
        assert_approx_eq!(0.0, hgbat(&vec![-1.0, -1.0, -1.0, -1.0]));
    }

    #[test]
    fn hgbat_not_optimum() {
        assert_ne!(0.0, hgbat(&vec![2.0, -1.3, 5.0]));
    }

    #[test]
    fn griewank_rosenbrock_optimum() {
        assert_approx_eq!(
            0.0,
            griewank_rosenbrock(&vec![1.0000027931852864, 1.0000049737406767])
        );
    }

    #[test]
    fn griewank_rosenbrock_not_optimum() {
        assert_ne!(0.0, griewank_rosenbrock(&vec![-1.0, -1.0]));
    }

    #[ignore]
    #[bench]
    fn bench_ackley(b: &mut Bencher) {
        let x = vec![1.0, 3.0];
        b.iter(|| {
            ackley(&x);
        });
    }

    #[ignore]
    #[bench]
    fn bench_rosenbrock(b: &mut Bencher) {
        let x = vec![1.0, 3.0];
        b.iter(|| {
            rosenbrock(&x);
        });
    }

}
