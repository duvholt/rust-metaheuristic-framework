fn uniform(female: &Vec<f64>, males: Vec<&Vec<f64>>, beta: f64) -> Vec<f64> {
    female
        .iter()
        .enumerate()
        .map(|(j, p1_j)| {
            beta * p1_j
                + males
                    .iter()
                    .map(|male| (1.0 - beta) / (males.len() as f64) * male[j])
                    .sum::<f64>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_crossover_single_male() {
        let female = vec![0.2, 0.4, 0.2];
        let male = vec![0.4, 0.2, 0.4];
        let beta = 0.5;

        let position = uniform(&female, vec![&male], beta);

        for value in position {
            assert_approx_eq!(value, 0.3);
        }
    }

    #[test]
    fn uniform_crossover_multiple_males() {
        let female = vec![0.2, 0.4, 0.2];
        let male1 = vec![0.4, 0.2, 0.4];
        let male2 = vec![0.5, 0.1, 0.4];
        let beta = 0.5;

        let position = uniform(&female, vec![&male1, &male2], beta);

        assert_eq!(position.len(), 3);
        assert_eq!(position[0], 0.325);
        assert_eq!(position[1], 0.275);
        assert_approx_eq!(position[2], 0.3);
    }
}
