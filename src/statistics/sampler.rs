use solution::{Solution, SolutionJSON};
use std::cell::RefCell;
use std::cmp::Ordering;

pub enum SamplerMode {
    LastGeneration,
    Evolution,
    EvolutionBest,
    FitnessSearch,
}

pub struct Sampler {
    mode: SamplerMode,
    samples: i64,
    max_iterations: i64,
    solutions: RefCell<Vec<SolutionJSON>>,
    generations: RefCell<Vec<Vec<SolutionJSON>>>,
}

impl Sampler {
    pub fn new(samples: i64, max_iterations: i64, mode: SamplerMode) -> Sampler {
        Sampler {
            samples,
            mode,
            max_iterations,
            solutions: RefCell::new(vec![]),
            generations: RefCell::new(vec![]),
        }
    }

    fn criteria_met(&self, iteration: i64) -> bool {
        match self.mode {
            SamplerMode::Evolution | SamplerMode::EvolutionBest => {
                if iteration == self.max_iterations {
                    return false;
                }
                if iteration % (self.max_iterations / self.samples) != 0 {
                    return false;
                }
                return true;
            }
            SamplerMode::LastGeneration => iteration == self.max_iterations,
            _ => false,
        }
    }

    fn find_best_solution<S: Solution<f64>>(&self, samples: &[S]) -> SolutionJSON {
        let best = samples
            .iter()
            .min_by(|a, b| {
                a.fitness()
                    .partial_cmp(&b.fitness())
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();
        SolutionJSON::from_single(best)
    }

    fn add_solution(&self, solution: SolutionJSON) {
        self.solutions.borrow_mut().push(solution);
    }

    pub fn population_sample_single<S: Solution<f64>>(&self, iteration: i64, samples: &[S]) {
        if !self.criteria_met(iteration) {
            return;
        }
        match self.mode {
            SamplerMode::EvolutionBest => {
                self.add_solution(self.find_best_solution(&samples));
            }
            SamplerMode::Evolution => {
                let mut solutions = samples
                    .iter()
                    .map(|s| SolutionJSON::from_single(s))
                    .collect();
                self.generations.borrow_mut().push(solutions);
            }
            SamplerMode::LastGeneration => {
                let mut solutions = samples
                    .iter()
                    .map(|s| SolutionJSON::from_single(s))
                    .collect();
                self.solutions.borrow_mut().append(&mut solutions);
            }
            _ => {}
        }
    }

    pub fn population_sample_multi<S: Solution<Vec<f64>>>(&self, iteration: i64, samples: &[S]) {
        if !self.criteria_met(iteration) {
            return;
        }
        let mut solutions: Vec<_> = samples
            .iter()
            .map(|sample| SolutionJSON::from_multi(sample))
            .collect();
        self.solutions.borrow_mut().append(&mut solutions);
    }

    fn sample_fitness(&self, solution: SolutionJSON) {
        match self.mode {
            SamplerMode::FitnessSearch => {
                self.add_solution(solution);
            }
            _ => {}
        }
    }

    pub fn sample_fitness_single(&self, fitness: &f64, position: &Vec<f64>) {
        self.sample_fitness(SolutionJSON {
            x: position.to_vec(),
            fitness: vec![*fitness],
        });
    }

    pub fn sample_fitness_multi(&self, fitness: &Vec<f64>, position: &Vec<f64>) {
        self.sample_fitness(SolutionJSON {
            x: position.to_vec(),
            fitness: fitness.to_vec(),
        });
    }

    pub fn solutions(&self) -> Vec<SolutionJSON> {
        match self.mode {
            SamplerMode::Evolution => self.generations
                .borrow()
                .iter()
                .cloned()
                .flat_map(|generation| generation)
                .collect(),
            _ => self.solutions.borrow().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solution::SingleTestSolution;

    #[test]
    fn samples_every_other_iteration() {
        let fitness: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0];
        let sampler = Sampler::new(5, fitness.len() as i64, SamplerMode::Evolution);

        for (iteration, fitness) in fitness.iter().enumerate() {
            sampler
                .population_sample_single(iteration as i64, &[SingleTestSolution::new(*fitness)]);
        }

        // Convert Solution back to fitness
        let sampler_fitness: Vec<_> = sampler.solutions().iter().map(|s| s.fitness[0]).collect();
        assert_eq!(
            sampler_fitness,
            vec![fitness[0], fitness[2], fitness[4], fitness[6], fitness[8]]
        );
    }

    #[test]
    fn samples_only_last_generation() {
        let generations = vec![
            vec![0.1, 0.2],
            vec![0.4, 0.5],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
            vec![0.9, 1.0],
        ];
        let sampler = Sampler::new(
            3,
            (generations.len() - 1) as i64,
            SamplerMode::LastGeneration,
        );

        for (iteration, generation) in generations.iter().enumerate() {
            let solutions: Vec<_> = generation
                .iter()
                .map(|fitness| SingleTestSolution::new(*fitness))
                .collect();
            sampler.population_sample_single(iteration as i64, &solutions);
        }

        let sampler_fitness: Vec<_> = sampler.solutions().iter().map(|s| s.fitness[0]).collect();
        assert_eq!(sampler_fitness, generations[4]);
    }

    #[test]
    fn samples_fitness_if_fitness_search() {
        let sampler = Sampler::new(0, 0, SamplerMode::FitnessSearch);

        assert_eq!(sampler.solutions().len(), 0);
        sampler.sample_fitness_single(&1.0, &vec![0.0, 0.1]);

        assert_eq!(sampler.solutions().len(), 1);
    }

    #[test]
    fn samples_fitness_if_fitness_search_multi() {
        let sampler = Sampler::new(0, 0, SamplerMode::FitnessSearch);

        assert_eq!(sampler.solutions().len(), 0);
        sampler.sample_fitness_multi(&vec![1.0, 2.0], &vec![0.0, 0.1]);

        assert_eq!(sampler.solutions().len(), 1);
    }

    #[test]
    fn does_not_samples_fitness_if_not_fitness_search() {
        let sampler = Sampler::new(0, 0, SamplerMode::Evolution);

        assert_eq!(sampler.solutions().len(), 0);
        sampler.sample_fitness_single(&1.0, &vec![0.0, 0.1]);

        assert_eq!(sampler.solutions().len(), 0);
    }

    #[test]
    fn only_saves_best() {
        let samples = [0.3, 0.2, 0.1, 0.4];
        let solutions: Vec<_> = samples
            .iter()
            .map(|fitness| SingleTestSolution::new(*fitness))
            .collect();
        let sampler = Sampler::new(10, 10, SamplerMode::EvolutionBest);

        for i in 0..10 {
            sampler.population_sample_single(i as i64, &solutions);

            let sampler_solutions = sampler.solutions();
            assert_eq!(sampler_solutions.len(), i + 1);
            assert_eq!(sampler_solutions[i].fitness, vec![0.1]);
        }
    }
}
