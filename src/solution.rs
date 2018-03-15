#[derive(Serialize)]
pub struct Solutions {
    pub solutions: Vec<Solution>,
    pub test_function: String,
}

#[derive(Serialize, Clone)]
pub struct Solution {
    pub x: Vec<f64>,
    pub fitness: f64,
}
impl Solution {
    pub fn new(x: Vec<f64>, fitness: f64) -> Solution {
        Solution { x, fitness }
    }
}
