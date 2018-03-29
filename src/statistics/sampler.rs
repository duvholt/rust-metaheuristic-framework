use solution::Solution;
use std::cell::RefCell;
use std::fmt::Debug;

pub enum SamplerMode {
    Test,
    LastGeneration,
    Evolution,
}

pub struct Sampler<S> {
    mode: SamplerMode,
    samples: usize,
    max_iterations: i64,
    solutions: RefCell<Vec<Box<Solution<S>>>>,
}

impl<S: Clone + Debug> Sampler<S> {
    pub fn new(samples: usize, max_iterations: i64, mode: SamplerMode) -> Sampler<S> {
        Sampler {
            samples,
            mode,
            max_iterations,
            solutions: RefCell::new(vec![]),
        }
    }

    pub fn iteration(&self, iteration: usize, sample: Box<Solution<S>>) {
        if iteration % (self.max_iterations as usize / self.samples) == 0 {
            self.solutions.borrow_mut().push(sample);
        }
    }
}

impl Sampler<f64> {
    pub fn solutions(&self) -> &RefCell<Vec<Box<Solution<f64>>>> {
        &self.solutions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solution::SingleTestSolution;

    #[test]
    fn samples_every_other_iteration() {
        let fitness: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0];
        let sampler: Sampler<f64> = Sampler::new(5, fitness.len() as i64, SamplerMode::Evolution);

        for (iteration, fitness) in fitness.iter().enumerate() {
            sampler.iteration(iteration, Box::new(SingleTestSolution::new(*fitness)));
        }

        // Convert Solution back to fitness
        let sampler_fitness: Vec<_> = sampler
            .solutions()
            .borrow()
            .iter()
            .map(|s| *s.fitness())
            .collect();
        assert_eq!(
            sampler_fitness,
            vec![fitness[0], fitness[2], fitness[4], fitness[6], fitness[8]]
        );
    }
}
