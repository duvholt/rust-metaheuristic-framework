pub fn dominates(a: &Vec<f64>, b: &Vec<f64>) -> bool {
    let mut equal = true;
    for i in 0..a.len() {
        if a[i] > b[i] {
            return false;
        } else if a[i] < b[i] {
            equal = false;
        }
    }
    return !equal;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominates_all() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.2, 0.3, 0.4];

        let a_dominates_b = dominates(&a, &b);

        assert!(a_dominates_b);
    }

    #[test]
    fn dominates_one() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.3, 0.3];

        let a_dominates_b = dominates(&a, &b);

        assert!(a_dominates_b);
    }

    #[test]
    fn dominates_equal() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.2, 0.3];

        let a_dominates_b = dominates(&a, &b);

        assert!(!a_dominates_b);
    }

    #[test]
    fn does_not_dominate() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.3, 0.1, 0.3];

        let a_dominates_b = dominates(&a, &b);

        assert!(!a_dominates_b);
    }
}
