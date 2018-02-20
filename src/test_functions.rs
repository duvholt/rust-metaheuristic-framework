use std::f64::consts;

pub fn ackley(x: f64, y: f64) -> f64 {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * consts::PI;
    return -a * (-b * (0.5 * (x.powf(2.0) + y.powf(2.0))).sqrt()).exp()
        - (0.5 * ((c * x).cos() + (c * y).cos())).exp() + a + consts::E;
}

pub fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0;
    let b = 100.0;
    (a - x).powf(2.0) + b * (y - x.powf(2.0)).powf(2.0)
}

pub fn zakharov(x: f64, y: f64) -> f64 {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;

    let dimensions = [x, y];
    let mut i = 1.0;
    dimensions.iter().for_each(|xi| {
        sum1 = sum1 + xi.powf(2.0);
        sum2 = sum2 + 0.5 * i * xi;
        i += 1.0;
    });
    return sum1 + sum2.powf(2.0) + sum2.powf(4.0);
}

pub fn himmelblau(x: f64, y: f64) -> f64 {
    return (x.powf(2.0) + y - 11.0).powf(2.0) + (x + y.powf(2.0) - 7.0).powf(2.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_optimum() {
        assert_eq!(0.0, rosenbrock(1.0, 1.0));
    }

    #[test]
    fn rosenbrock_not_optimum() {
        assert_ne!(0.0, rosenbrock(3.0, 2.0));
    }

    #[test]
    fn zakharov_optimum() {
        assert_eq!(0.0, zakharov(0.0, 0.0));
    }

    #[test]
    fn zakharov_not_optimum() {
        assert_ne!(0.0, zakharov(2.0, -1.3));
    }

    #[test]
    fn ackley_optimum() {
        // TODO: Check if this is actually the minimum
        assert_approx_eq!(0.0, ackley(0.0, 0.0));
    }

    #[test]
    fn ackley_not_optimum() {
        assert_ne!(0.0, ackley(2.0, -1.3));
    }

    #[test]
    fn himmelblau_optimum() {
        assert_eq!(0.0, himmelblau(3.0, 2.0));
        assert_approx_eq!(0.0, himmelblau(-2.805118, 3.131312));
        assert_approx_eq!(0.0, himmelblau(-3.779310, -3.283186));
        assert_approx_eq!(0.0, himmelblau(3.584428, -1.848126));
    }
}
