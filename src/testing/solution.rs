use solution::MultiTestSolution;

pub fn vec_to_multi_solution(solutions: Vec<Vec<f64>>) -> Vec<MultiTestSolution> {
    solutions
        .iter()
        .map(|solution| MultiTestSolution {
            position: solution.to_vec(),
            fitness: solution.to_vec(),
        })
        .collect()
}
