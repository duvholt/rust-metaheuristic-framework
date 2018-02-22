use rand::distributions::{IndependentSample, Range};
use rand;

pub struct Neighbourhood<'a> {
    dimonension: i32,
    space: f64,
    rng: rand::ThreadRng,
    test_function: &'a Fn(&Vec<f64>) -> f64,
}

impl<'a> Neighbourhood<'a> {
    pub fn new(
        dimonension: i32,
        space: f64,
        test_function: &'a Fn(&Vec<f64>) -> f64,
    ) -> Neighbourhood {
        return Neighbourhood {
            dimonension,
            space,
            rng: rand::thread_rng(),
            test_function,
        };
    }

    pub fn random_solution(&self) -> Solution {
        let between = Range::new(-self.space, self.space);
        let mut rng = rand::thread_rng();
        let x = (0..self.dimonension)
            .map(|_| between.ind_sample(&mut rng))
            .collect();
        let fitness = self.calculate_fitness(&x);
        Solution::new(x, fitness)
    }

    fn calculate_fitness(&self, x: &Vec<f64>) -> f64 {
        (self.test_function)(x)
    }

    fn single_dimension_neighbour(&mut self, x: &f64) -> f64 {
        let neighbour_space = 0.01;
        let between = Range::new(
            x - self.space * neighbour_space,
            x + self.space * neighbour_space,
        );
        between.ind_sample(&mut self.rng)
    }

    pub fn find(&mut self, solution: &Solution) -> Solution {
        let x = solution
            .x
            .iter()
            .map(|x| self.single_dimension_neighbour(x))
            .collect();
        let fitness = self.calculate_fitness(&x);
        Solution::new(x, fitness)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions;

    #[test]
    fn generates_neighbour() {
        let test_function = test_functions::rosenbrock;
        let mut neighbourhood = Neighbourhood::new(2, 1.0, &test_function);
        let solution = neighbourhood.random_solution();
        let neighbour = neighbourhood.find(&solution);

        let neighbour_x = neighbour.x[0];
        let neighbour_y = neighbour.x[1];
        let solution_x = neighbour.x[0];
        let solution_y = neighbour.x[1];
        assert!(neighbour_x < (solution_x + 1.0) && neighbour_x > (solution_x - 1.0));
        assert!(neighbour_y < (solution_y + 1.0) && neighbour_y > (solution_y - 1.0));
    }
}
