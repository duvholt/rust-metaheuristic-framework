pub fn multi_dummy(x: &Vec<f64>) -> Vec<f64> {
    x.to_vec()
}

pub fn single_dummy(x: &Vec<f64>) -> f64 {
    x.iter().map(|i| i).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_dummy_test() {
        assert_eq!(single_dummy(&vec![1.0, 1.1]), 2.1);
        assert_eq!(single_dummy(&vec![1.0, 1.0, 1.0]), 3.0);
        assert_eq!(single_dummy(&vec![0.0, 0.0]), 0.0);
    }
    #[test]
    fn multi_dummy_test() {
        assert_eq!(multi_dummy(&vec![1.0, 1.1]), [1.0, 1.1]);
    }
}
