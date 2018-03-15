#[derive(Serialize)]
pub struct Solutions {
    pub solutions: Vec<SolutionJSON>,
    pub test_function: String,
}

#[derive(Serialize, Clone)]
pub struct SolutionJSON {
    pub x: Vec<f64>,
    pub fitness: f64,
}

impl SolutionJSON {
    pub fn new(x: Vec<f64>, fitness: f64) -> SolutionJSON {
        SolutionJSON { x, fitness }
    }
}
