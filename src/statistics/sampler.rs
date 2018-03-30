use solution::{Solution, SolutionJSON};
use std::cell::RefCell;
use std::cmp::Ordering;

pub enum SamplerMode {
    Test,
    LastGeneration,
    Evolution,
}

pub struct Sampler {
    mode: SamplerMode,
    samples: i64,
    max_iterations: i64,
    solutions: RefCell<Vec<SolutionJSON>>,
}

impl Sampler {
    pub fn new(samples: i64, max_iterations: i64, mode: SamplerMode) -> Sampler {
        Sampler {
            samples,
            mode,
            max_iterations,
            solutions: RefCell::new(vec![]),
        }
    }

    pub fn criteria_met(&self, iteration: i64) -> bool {
        if iteration % (self.max_iterations / self.samples) != 0 {
            return false;
        }
        return true;
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

    pub fn iteration_single<S: Solution<f64>>(&self, iteration: i64, samples: &[S]) {
        if !self.criteria_met(iteration) {
            return;
        }
        self.solutions
            .borrow_mut()
            .push(self.find_best_solution(&samples))
    }

    pub fn iteration_multi<S: Solution<Vec<f64>>>(&self, iteration: i64, samples: &[S]) {
        if !self.criteria_met(iteration) {
            return;
        }
        let mut solutions: Vec<_> = samples
            .iter()
            .map(|sample| SolutionJSON::from_multi(sample))
            .collect();
        self.solutions.borrow_mut().append(&mut solutions);
    }

    pub fn solutions(&self) -> Vec<SolutionJSON> {
        self.solutions.borrow().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solution::SingleTestSolution;

    #[test]
    fn samples_every_other_iteration() {
        let fitness: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0];
        let sampler: Sampler = Sampler::new(5, fitness.len() as i64, SamplerMode::Evolution);

        for (iteration, fitness) in fitness.iter().enumerate() {
            sampler.iteration_single(iteration, &[SingleTestSolution::new(*fitness)]);
        }

        // Convert Solution back to fitness
        let sampler_fitness: Vec<_> = sampler.solutions().iter().map(|s| s.fitness[0]).collect();
        assert_eq!(
            sampler_fitness,
            vec![fitness[0], fitness[2], fitness[4], fitness[6], fitness[8]]
        );
    }
}
