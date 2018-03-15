use std::f64::consts;

pub fn cauchy(x: f64, tau: f64) -> f64 {
    1.0 / 2.0 + (1.0 / consts::PI) * (x / tau).atan()
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
}
