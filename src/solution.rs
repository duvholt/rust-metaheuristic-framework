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

pub trait Solution {
    fn position(&self) -> Vec<f64>;
    fn fitness(&self) -> f64;
}

pub trait MultiSolution {
    fn position(&self) -> &Vec<f64>;
    fn fitness(&self) -> &Vec<f64>;
}

pub fn solutions_to_json<S>(population: Vec<S>) -> Vec<SolutionJSON>
where
    S: Solution,
{
    let mut solutions: Vec<SolutionJSON> = population
        .iter()
        .map(|solution| SolutionJSON {
            x: solution.position(),
            fitness: vec![solution.fitness()],
        })
        .collect();
    solutions.sort_unstable_by(|a, b| {
        b.fitness[0]
            .partial_cmp(&a.fitness[0])
            .unwrap_or(Ordering::Equal)
    });
    solutions
}
