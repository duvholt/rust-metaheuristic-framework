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
    let a = 1.0;
    let b = 100.0;
    (0..x.len() / 2)
        .map(|i| {
            let i1 = 2 * i;
            let i2 = i1 + 1;
            b * (x[i1].powf(2.0) - x[i2]).powf(2.0) + (x[i1] - a).powf(2.0)
        })
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
    //f1
    result.push((1.0 + g) * 0.5 * (0..m - 1).map(|i| x[i]).product::<f64>());
    //f2 to fm-1
    for i in 1..m - 1 {
        let prod: f64 = (0..m - (i + 1)).map(|j| x[j]).product();
        result.push((1.0 + g) * 0.5 * prod * (1.0 - x[m - (i + 1)]));
    }
    //fm
    result.push((1.0 + g) * 0.5 * (1.0 - x[0]));
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
