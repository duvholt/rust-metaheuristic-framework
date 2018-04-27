#[cfg(windows)]
#[link(name = "msvcrt")]
extern "C" {
    fn tgamma(x: f64) -> f64;
}

#[cfg(not(windows))]
#[link(name = "m")]
extern "C" {
    fn tgamma(x: f64) -> f64;
}
fn gamma(x: f64) -> f64 {
    unsafe { tgamma(x) }
}

use rand::{weak_rng, Rng};
use std::f64::consts;

pub fn cauchy(x: f64, tau: f64) -> f64 {
    1.0 / 2.0 + (1.0 / consts::PI) * (x / tau).atan()
}

pub fn levy_flight(beta: f64) -> f64 {
    let numerator = gamma(1.0 + beta) * (consts::PI / 2.0).sin();
    let denominator = gamma((1.0 + beta) / 2.0) * beta * 2.0_f64.powf((beta - 1.0) / 2.0);
    let sigma = (numerator / denominator).powf(1.0 / beta);

    let mut rng = weak_rng();
    let u = rng.gen_range(0.0, sigma.powi(2));
    let v = rng.next_f64();

    (u / v.abs().powf(1.0 / beta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cauchy_lower() {
        let val2 = cauchy(-2.0, 1.0);

        assert_eq!(val2, 0.14758361765043326);
    }

    #[test]
    fn cauchy_top() {
        let val = cauchy(0.0, 1.0);

        assert_eq!(val, 0.5);
    }

    #[test]
    fn cauchy_higher() {
        let val = cauchy(2.0, 1.0);

        assert_eq!(val, 0.8524163823495667);
    }

    #[test]
    fn gamma_function_test() {
        assert_eq!(4.0 * 3.0 * 2.0 * 1.0, gamma(5.0));
    }
}
