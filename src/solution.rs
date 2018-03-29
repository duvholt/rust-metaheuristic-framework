use std::cmp::Ordering;

#[derive(Serialize)]
pub struct Solutions {
    pub solutions: Vec<SolutionJSON>,
    pub test_function: String,
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
}

pub trait Solution<F> {
    fn position(&self) -> &Vec<f64>;
    fn fitness(&self) -> &F;
}

pub fn solutions_to_json<S>(population: Vec<S>) -> Vec<SolutionJSON>
where
    S: Solution<f64>,
{
    let mut solutions: Vec<SolutionJSON> = population
        .iter()
        .map(|solution| SolutionJSON {
            x: solution.position().to_vec(),
            fitness: vec![solution.fitness().clone()],
        })
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
        .map(|solution| SolutionJSON {
            x: solution.position().to_vec(),
            fitness: solution.fitness().to_vec(),
        })
        .collect()
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
    fitness: Vec<f64>,
    position: Vec<f64>,
}

impl MultiTestSolution {
    pub fn new(fitness: Vec<f64>) -> MultiTestSolution {
        MultiTestSolution {
            position: fitness.to_vec(),
            fitness,
        }
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
