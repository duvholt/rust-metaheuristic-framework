use statistics::sampler::Sampler;
use std::cell::Cell;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

pub type SingleTestFunction = Fn(&Vec<f64>) -> f64;
pub type MultiTestFunction = Fn(&Vec<f64>) -> Vec<f64>;

pub type SingleTestFunctionVar = fn(&Vec<f64>) -> f64;
pub type MultiTestFunctionVar = fn(&Vec<f64>) -> Vec<f64>;
#[derive(Clone, Copy)]
pub enum TestFunctionVar {
    Single(SingleTestFunctionVar),
    Multi(MultiTestFunctionVar, &'static str),
}

pub fn get_single(
    test_function_var: TestFunctionVar,
) -> Result<SingleTestFunctionVar, &'static str> {
    match test_function_var {
        TestFunctionVar::Single(f) => Ok(f),
        _ => Err("Algorithm only supports single objective functions"),
    }
}

pub fn get_multi(
    test_function_var: TestFunctionVar,
) -> Result<(MultiTestFunctionVar, &'static str), &'static str> {
    match test_function_var {
        TestFunctionVar::Multi(f, s) => Ok((f, s)),
        _ => Err("Algorithm only supports multi objective functions"),
    }
}

pub struct FitnessEvaluator<'a, F: 'static>
where
    F: Clone,
{
    test_function: fn(&Vec<f64>) -> F,
    evaluations: Cell<i64>,
    max_evaluations: i64,
    pub sampler: &'a Sampler,
    shift_data: Option<Vec<f64>>,
    rotate_data: Option<Vec<f64>>,
    pub input_scale: f64,
    pub add_to_position: f64,
}

impl<'a, F: 'static> FitnessEvaluator<'a, F>
where
    F: Clone,
{
    pub fn new(
        test_function: fn(&Vec<f64>) -> F,
        max_evaluations: i64,
        sampler: &'a Sampler,
    ) -> FitnessEvaluator<F> {
        FitnessEvaluator {
            test_function,
            evaluations: Cell::new(0),
            max_evaluations,
            sampler,
            shift_data: None,
            rotate_data: None,
            input_scale: 1.0,
            add_to_position: 0.0,
        }
    }

    pub fn end_criteria(&self) -> bool {
        self.evaluations.get() >= self.max_evaluations
    }

    pub fn evaluations(&self) -> i64 {
        self.evaluations.get()
    }

    pub fn read_shifted(&mut self, number: usize, dimensions: usize) -> Result<(), Box<Error>> {
        let mut f = File::open(format!("input_data/shift_data_{}.txt", number))?;
        let mut contents = String::new();
        f.read_to_string(&mut contents)?;
        self.shift_data = Some(
            contents
                .split_whitespace()
                .take(dimensions)
                .map(|data| data.parse::<f64>().unwrap())
                .collect(),
        );
        Ok(())
    }

    pub fn read_rotate(&mut self, number: usize, dimensions: usize) -> Result<(), Box<Error>> {
        match dimensions {
            2 | 10 | 20 | 30 | 50 | 100 => (),
            _ => panic!("Rotate data does not exist for specified dimension"),
        }
        let mut f = File::open(format!("input_data/M_{}_D{}.txt", number, dimensions))?;
        let mut contents = String::new();
        f.read_to_string(&mut contents)?;
        self.rotate_data = Some(
            contents
                .split_whitespace()
                .take(dimensions * dimensions)
                .map(|data| data.parse::<f64>().unwrap())
                .collect(),
        );
        Ok(())
    }

    fn shift_input(&self, position: &Vec<f64>, shift_data: &Vec<f64>) -> Vec<f64> {
        position
            .iter()
            .zip(shift_data)
            .map(|(x_i, s_i)| x_i - s_i)
            .collect()
    }

    fn rotate_input(&self, position: &Vec<f64>, rotate_data: &Vec<f64>) -> Vec<f64> {
        let d = position.len();
        position
            .iter()
            .enumerate()
            .map(|(i, x_i)| (0..d).fold(0.0, |acc, j| acc + x_i * rotate_data[i * d + j]))
            .collect()
    }
}

impl<'a> FitnessEvaluator<'a, f64> {
    pub fn calculate_fitness(&self, position: &Vec<f64>) -> f64 {
        self.evaluations.set(self.evaluations.get() + 1);

        let shifted_position = self.shift_data
            .as_ref()
            .map(|shift_data| self.shift_input(&position, shift_data));
        let position = shifted_position.as_ref().unwrap_or(position);
        let scaled_position = position
            .into_iter()
            .map(|x_i| x_i * self.input_scale)
            .collect();
        let position = &scaled_position;
        let rotated_position = self.rotate_data
            .as_ref()
            .map(|rotate_data| self.rotate_input(&position, rotate_data));
        let position = rotated_position.as_ref().unwrap_or(position);
        let added_position = position
            .into_iter()
            .map(|x_i| x_i + self.add_to_position)
            .collect();
        let position = &added_position;
        let fitness = (self.test_function)(position);
        self.sampler.sample_fitness_single(&fitness, &position);
        fitness
    }
}

impl<'a> FitnessEvaluator<'a, Vec<f64>> {
    pub fn calculate_fitness(&self, position: &Vec<f64>) -> Vec<f64> {
        self.evaluations.set(self.evaluations.get() + 1);
        let fitness = (self.test_function)(position);
        self.sampler.sample_fitness_multi(&fitness, &position);
        fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::test_functions::multi_dummy;
    use testing::test_functions::single_dummy;
    use testing::utils::create_sampler_multi;

    #[test]
    fn fitness_evalator_calculates_single_fitness() {
        let sampler = create_sampler_multi();
        let fitness_evalator = FitnessEvaluator::new(single_dummy, 100, &sampler);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, 0.0);
    }

    #[test]
    fn fitness_evalator_calculates_multi_fitness() {
        let sampler = create_sampler_multi();
        let fitness_evalator = FitnessEvaluator::new(multi_dummy, 100, &sampler);

        let fitness = fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);

        assert_eq!(fitness, vec![0.0, 0.0]);
    }

    #[test]
    fn fitness_evalator_end_criteria() {
        let sampler = create_sampler_multi();
        let fitness_evalator = FitnessEvaluator::new(multi_dummy, 5, &sampler);

        assert_eq!(fitness_evalator.end_criteria(), false);
        for assert_value in vec![false, false, false, false, true, true, true] {
            fitness_evalator.calculate_fitness(&vec![0.0, 0.0]);
            assert_eq!(fitness_evalator.end_criteria(), assert_value);
        }

        assert_eq!(fitness_evalator.evaluations(), 7);
    }
}
