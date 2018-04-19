use std::f64::consts;

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

pub fn schaffer1(x: &Vec<f64>) -> Vec<f64> {
    vec![x[0].powf(2.0), (x[0] - 2.0).powf(2.0)]
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

pub fn dtlz1(x: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![];
    let m = 3;
    let k = x.len() - m + 1;
    let g: f64 = 100.0
        * ((x.len() - k) as f64
            + (k..x.len())
                .map(|i| (x[i] - 0.5).powi(2) - (20.0 * consts::PI * (x[i] - 0.5)).cos())
                .sum::<f64>());
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
    let mut result = vec![];
    let m = 3;
    let k = x.len() - m + 1;
    let g: f64 = (k..x.len()).map(|i| (x[i] - 0.5).powi(2)).sum();
    //f_1
    result.push(
        (1.0 + g)
            * (0..m - 1)
                .map(|i| (x[i] * consts::PI / 2.0).cos())
                .product::<f64>(),
    );
    //f_2 to f_m-1
    for i in 1..m - 1 {
        let product: f64 = (0..m - (i + 1))
            .map(|j| (x[j] * consts::PI / 2.0).cos())
            .product();
        result.push((1.0 + g) * product * (x[m - (i + 1)] * consts::PI / 2.0).sin());
    }
    //f_m
    result.push((1.0 + g) * (x[0] * consts::PI / 2.0).sin());
    result
}

pub fn dtlz3(x: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![];
    let m = 3;
    let k = x.len() - m + 1;
    let g: f64 = 100.0
        * ((x.len() - k) as f64
            + (k..x.len())
                .map(|i| (x[i] - 0.5).powi(2) - (20.0 * consts::PI * (x[i] - 0.5)).cos())
                .sum::<f64>());
    //f_1
    result.push(
        (1.0 + g)
            * (0..m - 1)
                .map(|i| (x[i] * consts::PI / 2.0).cos())
                .product::<f64>(),
    );
    //f_2 to f_m-1
    for i in 1..m - 1 {
        let product: f64 = (0..m - (i + 1))
            .map(|j| (x[j] * consts::PI / 2.0).cos())
            .product();
        result.push((1.0 + g) * product * (x[m - (i + 1)] * consts::PI / 2.0).sin());
    }
    //f_m
    result.push((1.0 + g) * (x[0] * consts::PI / 2.0).sin());
    result
}

pub fn dtlz4(x: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![];
    let m = 3;
    let a = 100;
    let k = x.len() - m + 1;
    let g: f64 = (k..x.len()).map(|i| (x[i] - 0.5).powi(2)).sum();
    //f_1
    result.push(
        (1.0 + g)
            * (0..m - 1)
                .map(|i| (x[i].powi(a) * consts::PI / 2.0).cos())
                .product::<f64>(),
    );
    //f_2 to f_m-1
    for i in 1..m - 1 {
        let product: f64 = (0..m - (i + 1))
            .map(|j| (x[j].powi(a) * consts::PI / 2.0).cos())
            .product();
        result.push((1.0 + g) * product * (x[m - (i + 1)].powi(a) * consts::PI / 2.0).sin());
    }
    //f_m
    result.push((1.0 + g) * (x[0].powi(a) * consts::PI / 2.0).sin());
    result
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
            .map(|(i, x_i)| (x_i / (i + 1) as f64).cos())
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
    (10.0 / d.powi(2))
        * x.iter()
            .enumerate()
            .map(|(i, x_i)| {
                (1.0 + (i as f64 + 1.0)
                    * (1..33)
                        .map(|j| {
                            let p = 2f64.powi(j) * x_i;
                            ((p - (p).floor()).abs()) / 2f64.powi(j)
                        })
                        .sum::<f64>())
                    .powf(10.0 / d.powf(1.2))
            })
            .product::<f64>() - 10.0 / d.powi(2)
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

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

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
