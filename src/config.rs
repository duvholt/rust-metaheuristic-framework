pub struct CommonConfig {
    pub verbose: bool,
    pub iterations: i64,
    pub evaluations: i64,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub dimensions: usize,
    pub population: usize,
}

pub struct AlgorithmInfo {
    pub number: usize,
    pub scale: f64,
    pub add: f64,
}
