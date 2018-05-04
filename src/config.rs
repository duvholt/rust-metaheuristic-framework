pub struct CommonConfig {
    pub verbose: u64,
    pub iterations: i64,
    pub evaluations: i64,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub dimensions: usize,
    pub population: usize,
    pub multi_upper_bound: Vec<f64>,
    pub multi_lower_bound: Vec<f64>,
}
