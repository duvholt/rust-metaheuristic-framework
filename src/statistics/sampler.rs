use solution::{Solution, SolutionJSON};
use std::cell::RefCell;

pub enum SamplerMode {
    Test,
    LastGeneration,
    Evolution,
}

pub struct Sampler {
    mode: SamplerMode,
    samples: usize,
    max_iterations: i64,
    solutions: RefCell<Vec<SolutionJSON>>,
}

impl Sampler {
    pub fn new(samples: usize, max_iterations: i64, mode: SamplerMode) -> Sampler {
        Sampler {
            samples,
            mode,
            max_iterations,
            solutions: RefCell::new(vec![]),
        }
    }

    pub fn criteria_met(&self, iteration: usize) -> bool {
        if iteration % (self.max_iterations as usize / self.samples) != 0 {
            return false;
        }
        return true;
    }

    pub fn iteration_single(&self, iteration: usize, sample: &Solution<f64>) {
        if !self.criteria_met(iteration) {
            return;
        }
        self.solutions
            .borrow_mut()
            .push(SolutionJSON::from_single(sample));
    }

    pub fn iteration_multi(&self, iteration: usize, sample: &Solution<Vec<f64>>) {
        if !self.criteria_met(iteration) {
            return;
        }
        self.solutions
            .borrow_mut()
            .push(SolutionJSON::from_multi(sample));
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
            sampler.iteration_single(iteration, &SingleTestSolution::new(*fitness));
        }

        // Convert Solution back to fitness
        let sampler_fitness: Vec<_> = sampler.solutions().iter().map(|s| s.fitness[0]).collect();
        assert_eq!(
            sampler_fitness,
            vec![fitness[0], fitness[2], fitness[4], fitness[6], fitness[8]]
        );
    }
}
