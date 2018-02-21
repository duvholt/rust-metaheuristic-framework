use rand::distributions::{IndependentSample, Range};
use rand;

pub struct Neighbourhood<'a> {
    space: f64,
    rng: rand::ThreadRng,
    test_function: &'a Fn(f64, f64) -> f64,
}

impl<'a> Neighbourhood<'a> {
    pub fn new(space: f64, test_function: &'a Fn(f64, f64) -> f64) -> Neighbourhood {
        return Neighbourhood {
            space,
            rng: rand::thread_rng(),
            test_function,
        };
    }

    pub fn random_solution(&self) -> Solution {
        let between = Range::new(-self.space, self.space);
        let mut rng = rand::thread_rng();
        let x = between.ind_sample(&mut rng);
        let y = between.ind_sample(&mut rng);
        Solution::new(x, y, self.calculate_fitness(x, y))
    }

    fn calculate_fitness(&self, x: f64, y: f64) -> f64 {
        (self.test_function)(x, y)
    }

    fn single_dimension_neighbour(&mut self, x: f64) -> f64 {
        let neighbour_space = 0.01;
        let between = Range::new(
            x - self.space * neighbour_space,
            x + self.space * neighbour_space,
        );
        between.ind_sample(&mut self.rng)
    }

    pub fn find(&mut self, solution: &Solution) -> Solution {
        let x = self.single_dimension_neighbour(solution.x);
        let y = self.single_dimension_neighbour(solution.y);
        Solution::new(x, y, self.calculate_fitness(x, y))
    }
}

#[derive(Serialize)]
pub struct Solutions {
    pub solutions: Vec<Solution>,
    pub test_function: String,
}

#[derive(Serialize, Clone, Copy)]
pub struct Solution {
    pub x: f64,
    pub y: f64,
    pub fitness: f64,
}
impl Solution {
    pub fn new(x: f64, y: f64, fitness: f64) -> Solution {
        Solution { x, y, fitness }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_functions;

    #[test]
    fn generates_neighbour() {
        let test_function = test_functions::rosenbrock;
        let mut neighbourhood = Neighbourhood::new(1.0, &test_function);
        let solution = neighbourhood.random_solution();
        let neighbour = neighbourhood.find(&solution);

        assert!(neighbour.x < (solution.x + 1.0) && neighbour.x > (solution.x - 1.0));
        assert!(neighbour.y < (solution.y + 1.0) && neighbour.y > (solution.y - 1.0));
    }
}
