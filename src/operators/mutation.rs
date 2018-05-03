use rand::Rng;

pub fn one_dimension(
    rng: &mut impl Rng,
    position: &Vec<f64>,
    lower_bound: f64,
    upper_bound: f64,
    iteration: i64,
    iterations: i64,
    mutation_rate: f64,
) -> Option<Vec<f64>> {
    let dimensions = position.len();
    let pm = (1.0 - (iteration as f64 / iterations as f64)).powf(1.0 / mutation_rate);
    if rng.gen::<f64>() >= pm {
        return None;
    }
    let diff_position = pm * (upper_bound - lower_bound);
    let j: usize = rng.gen_range(0, dimensions);

    let mut lb = position[j] - diff_position;
    if lb < lower_bound {
        lb = lower_bound;
    }

    let mut ub = position[j] + diff_position;
    if ub > upper_bound {
        ub = upper_bound;
    }
    if lb == ub {
        return None;
    }
    let x_j = rng.gen_range(lb, ub);
    let mut mutated_position = position.to_vec();
    mutated_position[j] = x_j;
    Some(mutated_position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::utils::create_rng;

    #[test]
    fn mutates_random_position() {
        let mut rng = create_rng();
        let position = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let new_position = one_dimension(&mut rng, &position, 0.0, 1.0, 0, 10, 1.0);

        assert_eq!(
            new_position,
            Some(vec![0.1, 0.21635575241586102, 0.3, 0.4, 0.5, 0.6])
        );
    }

    #[test]
    fn no_mutation_when_iteration_is_near_max_iterations() {
        let mut rng = create_rng();
        let position = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let new_position = one_dimension(&mut rng, &position, 0.0, 1.0, 999_999, 1_000_000, 0.1);

        assert_eq!(new_position, None);
    }
}
