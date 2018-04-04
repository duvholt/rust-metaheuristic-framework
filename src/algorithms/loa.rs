use fitness_evaluation::FitnessEvaluator;
use position::random_position;

struct Config {
    iterations: i64,
    population: usize,
    upper_bound: f64,
    lower_bound: f64,
    dimension: usize,
}

struct Lion {
    position: Vec<f64>,
    fitness: f64,
}

fn random_population(config: &Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<Lion> {
    (0..config.population)
        .map(|_| {
            let position =
                random_position(config.lower_bound, config.upper_bound, config.dimension);
            let fitness = fitness_evaluator.calculate_fitness(&position);
            Lion { position, fitness }
        })
        .collect()
}

fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) {
    let population = random_population(&config, &fitness_evaluator);
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::utils::{create_evaluator, create_sampler};

    fn create_config() -> Config {
        Config {
            iterations: 100,
            population: 10,
            upper_bound: 1.0,
            lower_bound: -1.0,
            dimension: 2,
        }
    }
    #[test]
    fn creates_population() {
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        let config = create_config();

        let population = random_population(&config, &fitness_evaluator);

        assert_eq!(population.len(), config.population);
    }
}
