pub fn uniform(female: &Vec<f64>, males: &Vec<&Vec<f64>>, beta: f64) -> (Vec<f64>, Vec<f64>) {
    (
        uniform_single_offspring(&female, &males, beta),
        uniform_single_offspring(&female, &males, 1.0 - beta),
    )
}

fn uniform_single_offspring(female: &Vec<f64>, males: &Vec<&Vec<f64>>, beta: f64) -> Vec<f64> {
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

        let position = uniform_single_offspring(&female, &vec![&male], beta);

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

        let position = uniform_single_offspring(&female, &vec![&male1, &male2], beta);

        assert_eq!(position.len(), 3);
        assert_eq!(position[0], 0.325);
        assert_eq!(position[1], 0.275);
        assert_approx_eq!(position[2], 0.3);
    }

    #[test]
    fn uniform_produces_two_offsprings() {
        let female = vec![0.2, 0.4, 0.2];
        let male1 = vec![0.4, 0.2, 0.4];
        let male2 = vec![0.5, 0.1, 0.4];
        let beta = 0.4;

        let (offspring1, offspring2) = uniform(&female, &vec![&male1, &male2], beta);

        assert_eq!(offspring1, vec![0.35000000000000003, 0.25, 0.32]);
        assert_eq!(offspring2, vec![0.30000000000000004, 0.3, 0.28]);
    }
}
