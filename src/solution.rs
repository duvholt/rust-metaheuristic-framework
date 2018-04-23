use ordered_float::NotNaN;
use std::cmp::Ordering;
use std::hash;

#[derive(Clone)]
pub enum Objective {
    Single,
    Multi,
}

#[derive(Serialize)]
pub struct Solutions {
    pub solutions: Vec<SolutionJSON>,
    pub test_function: String,
    pub plot_bounds: bool,
    pub upper_bound: f64,
    pub lower_bound: f64,
}

#[derive(Serialize, Clone)]
pub struct SolutionJSON {
    pub x: Vec<f64>,
    pub fitness: Vec<f64>,
}

impl SolutionJSON {
    pub fn new(x: Vec<f64>, fitness: Vec<f64>) -> SolutionJSON {
        SolutionJSON { x, fitness }
    }

    pub fn from_single(solution: &Solution<f64>) -> SolutionJSON {
        SolutionJSON {
            x: solution.position().to_vec(),
            fitness: vec![*solution.fitness()],
        }
    }

    pub fn from_multi(solution: &Solution<Vec<f64>>) -> SolutionJSON {
        SolutionJSON {
            x: solution.position().to_vec(),
            fitness: solution.fitness().to_vec(),
        }
    }
}

pub trait Solution<F> {
    fn position(&self) -> &Vec<f64>;
    fn fitness(&self) -> &F;
    fn position_to_notnan(&self) -> Vec<NotNaN<f64>> {
        self.position()
            .iter()
            .map(|value| NotNaN::from(*value))
            .collect()
    }
}

pub fn solutions_to_json<S>(population: Vec<S>) -> Vec<SolutionJSON>
where
    S: Solution<f64>,
{
    let mut solutions: Vec<SolutionJSON> = population
        .iter()
        .map(|s| SolutionJSON::from_single(s))
        .collect();
    solutions.sort_unstable_by(|a, b| {
        b.fitness[0]
            .partial_cmp(&a.fitness[0])
            .unwrap_or(Ordering::Equal)
    });
    solutions
}

pub fn multi_solutions_to_json<M>(population: Vec<M>) -> Vec<SolutionJSON>
where
    M: Solution<Vec<f64>>,
{
    population
        .iter()
        .map(|s| SolutionJSON::from_multi(s))
        .collect()
}

pub fn sort_solutions_by_fitness(solutions: &mut [impl Solution<f64>]) {
    solutions.sort_unstable_by(|a, b| {
        a.fitness()
            .partial_cmp(&b.fitness())
            .unwrap_or(Ordering::Equal)
    })
}

// Structs used for testing
#[derive(Clone, Debug)]
pub struct SingleTestSolution {
    fitness: f64,
    position: Vec<f64>,
}

impl SingleTestSolution {
    pub fn new(fitness: f64) -> SingleTestSolution {
        SingleTestSolution {
            position: vec![fitness, fitness],
            fitness,
        }
    }
}

impl PartialEq for SingleTestSolution {
    fn eq(&self, other: &SingleTestSolution) -> bool {
        self.fitness == other.fitness
    }
}

impl Solution<f64> for SingleTestSolution {
    fn position(&self) -> &Vec<f64> {
        &self.position
    }

    fn fitness(&self) -> &f64 {
        &self.fitness
    }
}

#[derive(Clone, Debug)]
pub struct MultiTestSolution {
    pub fitness: Vec<f64>,
    pub position: Vec<f64>,
}

impl MultiTestSolution {
    pub fn new(fitness: Vec<f64>) -> MultiTestSolution {
        MultiTestSolution {
            position: fitness.to_vec(),
            fitness,
        }
    }
}

impl PartialEq for MultiTestSolution {
    fn eq(&self, other: &MultiTestSolution) -> bool {
        self.position == other.position
    }
}

impl Eq for MultiTestSolution {}

impl hash::Hash for MultiTestSolution {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.position_to_notnan().hash(state)
    }
}

impl Solution<Vec<f64>> for MultiTestSolution {
    fn position(&self) -> &Vec<f64> {
        &self.position
    }

    fn fitness(&self) -> &Vec<f64> {
        &self.fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_population_by_fitness_test() {
        let mut population = vec![
            SingleTestSolution {
                position: vec![4.0, 2.6],
                fitness: 1000.0,
            },
            SingleTestSolution {
                position: vec![3.0, 2.3],
                fitness: 100.0,
            },
            SingleTestSolution {
                position: vec![2.0, 2.1],
                fitness: 10.0,
            },
            SingleTestSolution {
                position: vec![1.0, 2.0],
                fitness: 1.0,
            },
        ];
        let mut population_clone = population.clone();
        population_clone.swap(0, 3);
        population_clone.swap(2, 1);
        sort_solutions_by_fitness(&mut population);
        assert_eq!(population, population_clone);
    }
}
