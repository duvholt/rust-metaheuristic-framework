use ansi_term::Color::{Blue, Cyan, Green, Red, Yellow};
use itertools::Itertools;
use itertools::MinMaxResult;
use solution::{Objective, Solution, SolutionJSON};
use statistical::{mean, population_standard_deviation};
use statistics::fronts::{front_min_max, normalize_front};
use statistics::measure::{gd, hyper_volume, igd};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::io::Write;

#[derive(Clone)]
pub enum SamplerMode {
    LastGeneration,
    Evolution,
    EvolutionBest,
    FitnessSearch,
}

#[derive(Serialize, Clone)]
pub struct SamplerJSON {
    mean: f64,
    standard_deviation: f64,
    pub test_function: String,
}

pub struct Sampler {
    mode: SamplerMode,
    pub objective: Objective,
    samples: i64,
    max_iterations: i64,
    solutions: RefCell<Vec<SolutionJSON>>,
    generations: RefCell<Vec<Vec<SolutionJSON>>>,
    pareto_front: Option<Vec<Vec<f64>>>,
    min_max_front: Option<Vec<(f64, f64)>>,
    runs: RefCell<Vec<f64>>,
}

impl Sampler {
    pub fn new(
        samples: i64,
        max_iterations: i64,
        mode: SamplerMode,
        objective: Objective,
    ) -> Sampler {
        Sampler {
            samples,
            mode,
            max_iterations,
            solutions: RefCell::new(vec![]),
            generations: RefCell::new(vec![]),
            objective,
            pareto_front: None,
            min_max_front: None,
            runs: RefCell::new(vec![]),
        }
    }

    pub fn set_pareto_front(&mut self, pareto_front: Vec<Vec<f64>>) {
        let min_max = front_min_max(&pareto_front);
        let pareto_front = normalize_front(&pareto_front, &min_max);
        self.min_max_front = Some(min_max);
        self.pareto_front = Some(pareto_front);
    }

    fn criteria_met(&self, iteration: i64) -> bool {
        match self.mode {
            SamplerMode::Evolution | SamplerMode::EvolutionBest => {
                if iteration == self.max_iterations {
                    return false;
                }
                if iteration % (self.max_iterations / self.samples) != 0 {
                    return false;
                }
                return true;
            }
            SamplerMode::LastGeneration => iteration == self.max_iterations,
            _ => false,
        }
    }

    fn find_best_solution<S: Solution<f64>>(&self, samples: &[S]) -> SolutionJSON {
        let best = samples
            .iter()
            .min_by(|a, b| {
                a.fitness()
                    .partial_cmp(&b.fitness())
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();
        SolutionJSON::from_single(best)
    }

    fn add_solution(&self, solution: SolutionJSON) {
        self.solutions.borrow_mut().push(solution);
    }

    pub fn population_sample_single<S: Solution<f64>>(&self, iteration: i64, samples: &[S]) {
        if !self.criteria_met(iteration) {
            return;
        }
        match self.mode {
            SamplerMode::EvolutionBest => {
                self.add_solution(self.find_best_solution(&samples));
            }
            SamplerMode::Evolution => {
                let mut solutions = samples
                    .iter()
                    .map(|s| SolutionJSON::from_single(s))
                    .collect();
                self.generations.borrow_mut().push(solutions);
            }
            SamplerMode::LastGeneration => {
                let mut solutions = samples
                    .iter()
                    .map(|s| SolutionJSON::from_single(s))
                    .collect();
                self.solutions.borrow_mut().append(&mut solutions);
            }
            _ => {}
        }
    }

    pub fn population_sample_multi<S: Solution<Vec<f64>>>(&self, iteration: i64, samples: &[S]) {
        if !self.criteria_met(iteration) {
            return;
        }
        let mut solutions: Vec<_> = samples
            .iter()
            .map(|sample| SolutionJSON::from_multi(sample))
            .collect();
        match self.mode {
            SamplerMode::Evolution => {
                self.generations.borrow_mut().push(solutions);
            }
            SamplerMode::LastGeneration => {
                self.solutions.borrow_mut().append(&mut solutions);
            }
            _ => {}
        }
    }

    fn sample_fitness(&self, solution: SolutionJSON) {
        match self.mode {
            SamplerMode::FitnessSearch => {
                self.add_solution(solution);
            }
            _ => {}
        }
    }

    pub fn sample_fitness_single(&self, fitness: &f64, position: &Vec<f64>) {
        self.sample_fitness(SolutionJSON {
            x: position.to_vec(),
            fitness: vec![*fitness],
        });
    }

    pub fn sample_fitness_multi(&self, fitness: &Vec<f64>, position: &Vec<f64>) {
        self.sample_fitness(SolutionJSON {
            x: position.to_vec(),
            fitness: fitness.to_vec(),
        });
    }

    pub fn solutions(&self) -> Vec<SolutionJSON> {
        match self.mode {
            SamplerMode::Evolution => self.generations
                .borrow()
                .iter()
                .cloned()
                .flat_map(|generation| generation)
                .collect(),
            _ => self.solutions.borrow().clone(),
        }
    }

    fn print_mean_and_stddev(mut writer: impl Write, values: &Vec<f64>) {
        write!(
            writer,
            "Average {} Standard deviation {}\n",
            Green.paint(format!("{:10.4e}", mean(&values))),
            Green.paint(format!(
                "{:10.4e}",
                population_standard_deviation(&values, None)
            )),
        ).unwrap();
    }

    fn print_best_position(mut writer: impl Write, solutions: &Vec<SolutionJSON>) {
        let best = solutions
            .iter()
            .min_by(|a, b| a.fitness[0].partial_cmp(&b.fitness[0]).unwrap())
            .unwrap();
        write!(
            writer,
            "Best position: {}\n",
            Yellow.paint(format!("{:?}", best.x))
        ).unwrap();
    }

    fn print_min_max(mut writer: impl Write, values: &Vec<f64>) {
        let minmax = values.iter().minmax();
        if let MinMaxResult::MinMax(min, max) = minmax {
            write!(
                writer,
                "Min: {}. Max: {}\n",
                Cyan.paint(format!("{:10.4e}", min)),
                Red.paint(format!("{:10.4e}", max)),
            ).unwrap();
        }
    }

    fn calculate_igd(&self, front: &Vec<Vec<f64>>) -> f64 {
        let front = normalize_front(&front, &self.min_max_front.clone().unwrap());
        igd(&front, &self.pareto_front.clone().unwrap())
    }

    fn calculate_gd(&self, front: &Vec<Vec<f64>>) -> f64 {
        let front = normalize_front(&front, &self.min_max_front.clone().unwrap());
        gd(&front, &self.pareto_front.clone().unwrap())
    }

    fn calculate_hv(&self, front: &Vec<Vec<f64>>) -> f64 {
        let front = normalize_front(&front, &self.min_max_front.clone().unwrap());
        hyper_volume(&front)
    }

    fn print_multi_measures(&self, mut writer: impl Write, generation: &Vec<SolutionJSON>) {
        let front = generation
            .iter()
            .map(|solution| solution.fitness.clone())
            .collect();
        let igd_value = self.calculate_igd(&front);
        let gd_value = self.calculate_gd(&front);
        let hv_value = self.calculate_hv(&front);
        write!(
            &mut writer,
            "IGD: {}\n\
             GD: {}\n\
             HV: {}\n",
            Green.paint(format!("{:10.4e}", igd_value)),
            Green.paint(format!("{:10.4e}", gd_value)),
            Green.paint(format!("{:10.4e}", hv_value)),
        ).unwrap();
    }

    fn print_evolution(&self, mut writer: impl Write) {
        for (i, generation) in self.generations.borrow().iter().enumerate() {
            // Prefix with generation index
            write!(&mut writer, "[{:2}] ", i).unwrap();
            match self.objective {
                Objective::Single => {
                    let fitness_values: Vec<_> = generation
                        .iter()
                        .map(|solution| solution.fitness[0])
                        .collect();
                    Sampler::print_mean_and_stddev(&mut writer, &fitness_values);
                    Sampler::print_min_max(&mut writer, &fitness_values);
                    Sampler::print_best_position(&mut writer, &generation);
                }
                Objective::Multi => {
                    self.print_multi_measures(&mut writer, &generation);
                }
            }
        }
    }

    fn print_last_generation(&self, mut writer: impl Write) {
        match self.objective {
            Objective::Single => {
                let fitness_values: Vec<_> = self.solutions
                    .borrow()
                    .iter()
                    .map(|solution| solution.fitness[0])
                    .collect();
                Sampler::print_mean_and_stddev(&mut writer, &fitness_values);
                Sampler::print_min_max(&mut writer, &fitness_values);
                Sampler::print_best_position(&mut writer, &self.solutions.borrow());
            }
            Objective::Multi => {
                self.print_multi_measures(&mut writer, &self.solutions.borrow());
            }
        }
    }

    fn print_evolution_best(&self, mut writer: impl Write) {
        for (i, solution) in self.solutions.borrow().iter().enumerate() {
            write!(
                &mut writer,
                "[{:2}] Fitness: {}\n",
                i,
                Green.paint(format!("{:10.4e}", solution.fitness[0]))
            ).unwrap();
        }
        Sampler::print_best_position(&mut writer, &self.solutions.borrow());
    }

    fn print_fitness_search(&self, mut writer: impl Write) {
        let fitness_values: Vec<_> = self.solutions
            .borrow()
            .iter()
            .map(|solution| solution.fitness[0])
            .collect();
        Sampler::print_mean_and_stddev(&mut writer, &fitness_values);
        Sampler::print_min_max(&mut writer, &fitness_values);
        Sampler::print_best_position(&mut writer, &self.solutions.borrow());
    }

    pub fn print_run_statistics(&self, mut writer: impl Write) {
        println!("{}", Yellow.underline().paint("Sample Statistics"));
        match self.mode {
            SamplerMode::Evolution => {
                write!(
                    &mut writer,
                    "Mode: Evolution with {} samples\n",
                    self.samples
                ).unwrap();
                self.print_evolution(&mut writer);
            }
            SamplerMode::LastGeneration => {
                write!(&mut writer, "Mode: Last Generation\n").unwrap();
                self.print_last_generation(&mut writer);
            }
            SamplerMode::EvolutionBest => {
                write!(
                    &mut writer,
                    "Mode: Best Solution Evolution with {} samples\n",
                    self.samples
                ).unwrap();
                self.print_evolution_best(&mut writer);
            }
            SamplerMode::FitnessSearch => {
                write!(&mut writer, "Mode: All fitness evaluations\n").unwrap();
                self.print_fitness_search(&mut writer);
            }
        }
        println!("{}", Yellow.underline().paint("End Sample Statistics"));
    }

    fn best_fitness(&self) -> f64 {
        let solutions = self.solutions();
        match self.objective {
            Objective::Single => solutions
                .iter()
                .map(|s| s.fitness[0])
                .min_by(|a, b| a.partial_cmp(&b).unwrap())
                .unwrap(),
            Objective::Multi => {
                let front = solutions
                    .iter()
                    .map(|solution| solution.fitness.clone())
                    .collect();
                self.calculate_hv(&front)
            }
        }
    }

    pub fn save_run(&self) {
        let best_fitness = self.best_fitness();
        self.runs.borrow_mut().push(best_fitness);
    }

    pub fn end_run(&self) {
        self.solutions.borrow_mut().clear();
        self.generations.borrow_mut().clear();
    }

    pub fn print_statistics(&self, mut writer: impl Write) {
        println!("{}", Blue.underline().paint("Run Statistics"));
        let runs = self.runs.borrow().to_vec();
        write!(&mut writer, "Number of runs: {}\n", runs.len()).unwrap();
        Sampler::print_mean_and_stddev(&mut writer, &runs);
        Sampler::print_min_max(&mut writer, &runs);
        println!("{}", Blue.underline().paint("End Run Statistics"));
    }

    pub fn to_json(&self) -> SamplerJSON {
        let runs = self.runs.borrow().to_vec();
        SamplerJSON {
            mean: mean(&runs),
            standard_deviation: population_standard_deviation(&runs, None),
            test_function: "".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solution::{MultiTestSolution, SingleTestSolution};

    #[test]
    fn samples_every_other_iteration() {
        let fitness: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0];
        let sampler = Sampler::new(
            5,
            fitness.len() as i64,
            SamplerMode::Evolution,
            Objective::Single,
        );

        for (iteration, fitness) in fitness.iter().enumerate() {
            sampler
                .population_sample_single(iteration as i64, &[SingleTestSolution::new(*fitness)]);
        }

        // Convert Solution back to fitness
        let sampler_fitness: Vec<_> = sampler.solutions().iter().map(|s| s.fitness[0]).collect();
        assert_eq!(
            sampler_fitness,
            vec![fitness[0], fitness[2], fitness[4], fitness[6], fitness[8]]
        );
    }

    #[test]
    fn samples_every_other_iteration_multi() {
        let fitness: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0];
        let sampler = Sampler::new(
            5,
            fitness.len() as i64,
            SamplerMode::Evolution,
            Objective::Multi,
        );

        for (iteration, fitness) in fitness.iter().enumerate() {
            sampler.population_sample_multi(
                iteration as i64,
                &[MultiTestSolution::new(vec![*fitness, *fitness])],
            );
        }

        // Convert Solution back to fitness
        let sampler_fitness: Vec<_> = sampler.solutions().iter().map(|s| s.fitness[0]).collect();
        assert_eq!(
            sampler_fitness,
            vec![fitness[0], fitness[2], fitness[4], fitness[6], fitness[8]]
        );
    }

    #[test]
    fn samples_only_last_generation() {
        let generations = vec![
            vec![0.1, 0.2],
            vec![0.4, 0.5],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
            vec![0.9, 1.0],
        ];
        let sampler = Sampler::new(
            3,
            (generations.len() - 1) as i64,
            SamplerMode::LastGeneration,
            Objective::Single,
        );

        for (iteration, generation) in generations.iter().enumerate() {
            let solutions: Vec<_> = generation
                .iter()
                .map(|fitness| SingleTestSolution::new(*fitness))
                .collect();
            sampler.population_sample_single(iteration as i64, &solutions);
        }

        let sampler_fitness: Vec<_> = sampler.solutions().iter().map(|s| s.fitness[0]).collect();
        assert_eq!(sampler_fitness, generations[4]);
    }

    #[test]
    fn samples_only_last_generation_multi() {
        let generations = vec![
            vec![vec![0.1, 0.1], vec![0.2, 0.2]],
            vec![vec![0.4, 0.4], vec![0.5, 0.5]],
            vec![vec![0.5, 0.5], vec![0.6, 0.6]],
            vec![vec![0.7, 0.7], vec![0.8, 0.8]],
            vec![vec![0.9, 0.9], vec![1.0, 1.0]],
        ];
        let sampler = Sampler::new(
            3,
            (generations.len() - 1) as i64,
            SamplerMode::LastGeneration,
            Objective::Multi,
        );

        for (iteration, generation) in generations.iter().enumerate() {
            let solutions: Vec<_> = generation
                .iter()
                .map(|fitness| MultiTestSolution::new(fitness.clone()))
                .collect();
            sampler.population_sample_multi(iteration as i64, &solutions);
        }

        let sampler_fitness: Vec<_> = sampler
            .solutions()
            .iter()
            .map(|s| s.fitness.clone())
            .collect();
        assert_eq!(sampler_fitness, generations[4]);
    }

    #[test]
    fn samples_fitness_if_fitness_search() {
        let sampler = Sampler::new(0, 0, SamplerMode::FitnessSearch, Objective::Single);

        assert_eq!(sampler.solutions().len(), 0);
        sampler.sample_fitness_single(&1.0, &vec![0.0, 0.1]);

        assert_eq!(sampler.solutions().len(), 1);
    }

    #[test]
    fn samples_fitness_if_fitness_search_multi() {
        let sampler = Sampler::new(0, 0, SamplerMode::FitnessSearch, Objective::Multi);

        assert_eq!(sampler.solutions().len(), 0);
        sampler.sample_fitness_multi(&vec![1.0, 2.0], &vec![0.0, 0.1]);

        assert_eq!(sampler.solutions().len(), 1);
    }

    #[test]
    fn does_not_samples_fitness_if_not_fitness_search() {
        let sampler = Sampler::new(0, 0, SamplerMode::Evolution, Objective::Single);

        assert_eq!(sampler.solutions().len(), 0);
        sampler.sample_fitness_single(&1.0, &vec![0.0, 0.1]);

        assert_eq!(sampler.solutions().len(), 0);
    }

    #[test]
    fn only_saves_best() {
        let samples = [0.3, 0.2, 0.1, 0.4];
        let solutions: Vec<_> = samples
            .iter()
            .map(|fitness| SingleTestSolution::new(*fitness))
            .collect();
        let sampler = Sampler::new(10, 10, SamplerMode::EvolutionBest, Objective::Single);

        for i in 0..10 {
            sampler.population_sample_single(i as i64, &solutions);

            let sampler_solutions = sampler.solutions();
            assert_eq!(sampler_solutions.len(), i + 1);
            assert_eq!(sampler_solutions[i].fitness, vec![0.1]);
        }
    }

    fn create_solutions() -> Vec<SingleTestSolution> {
        let samples = [0.3, 0.2, 0.1, 0.4];
        samples
            .iter()
            .map(|fitness| SingleTestSolution::new(*fitness))
            .collect()
    }

    fn create_solutions_multi() -> Vec<MultiTestSolution> {
        let samples = [0.3, 0.2, 0.1, 0.4];
        samples
            .iter()
            .map(|fitness| MultiTestSolution::new(vec![*fitness, *fitness]))
            .collect()
    }

    #[test]
    fn prints_evolution() {
        let solutions = create_solutions();
        let sampler = Sampler::new(10, 10, SamplerMode::Evolution, Objective::Single);

        sampler.population_sample_single(0, &solutions);
        let mut output = Vec::new();
        sampler.print_run_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Mode: Evolution with 10 samples\n\
                 [ 0] Average {} Standard deviation {}\n\
                 Min: {}. Max: {}\n\
                 Best position: {}\n",
                Green.paint(" 2.5000e-1"),
                Green.paint(" 1.1180e-1"),
                Cyan.paint(" 1.0000e-1"),
                Red.paint(" 4.0000e-1"),
                Yellow.paint("[0.1, 0.1]"),
            )
        );
    }

    #[test]
    fn prints_evolution_multi() {
        let solutions = create_solutions_multi();
        let mut sampler = Sampler::new(10, 10, SamplerMode::Evolution, Objective::Multi);
        sampler.set_pareto_front(vec![vec![0.25, 0.25], vec![0.15, 0.15], vec![0.05, 0.05]]);

        sampler.population_sample_multi(0, &solutions);
        let mut output = Vec::new();
        sampler.print_run_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Mode: Evolution with 10 samples\n\
                 [ 0] IGD: {}\n\
                 GD: {}\n\
                 HV: {}\n",
                Green.paint(" 2.0412e-1"),
                Green.paint(" 3.0619e-1"),
                Green.paint(" 5.6250e-1"),
            )
        );
    }

    #[test]
    fn prints_best() {
        let solutions = create_solutions();
        let sampler = Sampler::new(10, 10, SamplerMode::EvolutionBest, Objective::Single);

        sampler.population_sample_single(0, &solutions);
        let mut output = Vec::new();
        sampler.print_run_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Mode: Best Solution Evolution with 10 samples\n\
                 [ 0] Fitness: {}\n\
                 Best position: {}\n",
                Green.paint(" 1.0000e-1"),
                Yellow.paint("[0.1, 0.1]")
            )
        );
    }

    #[test]
    fn prints_last() {
        let solutions = create_solutions();
        let sampler = Sampler::new(10, 10, SamplerMode::LastGeneration, Objective::Single);

        sampler.population_sample_single(10, &solutions);
        let mut output = Vec::new();
        sampler.print_run_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Mode: Last Generation\n\
                 Average {} Standard deviation {}\n\
                 Min: {}. Max: {}\n\
                 Best position: {}\n",
                Green.paint(" 2.5000e-1"),
                Green.paint(" 1.1180e-1"),
                Cyan.paint(" 1.0000e-1"),
                Red.paint(" 4.0000e-1"),
                Yellow.paint("[0.1, 0.1]")
            )
        );
    }

    #[test]
    fn prints_last_multi() {
        let solutions = create_solutions_multi();
        let mut sampler = Sampler::new(10, 10, SamplerMode::LastGeneration, Objective::Multi);
        sampler.set_pareto_front(vec![vec![0.25, 0.25], vec![0.15, 0.15], vec![0.05, 0.05]]);

        sampler.population_sample_multi(10, &solutions);
        let mut output = Vec::new();
        sampler.print_run_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Mode: Last Generation\n\
                 IGD: {}\n\
                 GD: {}\n\
                 HV: {}\n",
                Green.paint(" 2.0412e-1"),
                Green.paint(" 3.0619e-1"),
                Green.paint(" 5.6250e-1"),
            )
        );
    }

    #[test]
    fn prints_fitness() {
        let sampler = Sampler::new(10, 10, SamplerMode::FitnessSearch, Objective::Single);

        sampler.sample_fitness_single(&2.0, &vec![0.0, 0.1]);
        sampler.sample_fitness_single(&1.0, &vec![0.1, 0.1]);
        sampler.sample_fitness_single(&3.0, &vec![0.2, 0.1]);
        let mut output = Vec::new();
        sampler.print_run_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Mode: All fitness evaluations\n\
                 Average {} Standard deviation {}\n\
                 Min: {}. Max: {}\n\
                 Best position: {}\n",
                Green.paint("  2.0000e0"),
                Green.paint(" 8.1650e-1"),
                Cyan.paint("  1.0000e0"),
                Red.paint("  3.0000e0"),
                Yellow.paint("[0.1, 0.1]")
            )
        );
    }

    #[test]
    fn prints_fitness_multi() {
        let solutions = create_solutions_multi();
        let fitness_values = vec![2.0, 1.0, 3.0];
        let sampler = Sampler::new(10, 10, SamplerMode::FitnessSearch, Objective::Multi);

        for (solution, fitness) in solutions[..3].iter().zip(fitness_values) {
            sampler.sample_fitness_multi(&vec![fitness, fitness], &solution.fitness());
        }
        let mut output = Vec::new();
        sampler.print_run_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Mode: All fitness evaluations\n\
                 Average {} Standard deviation {}\n\
                 Min: {}. Max: {}\n\
                 Best position: {}\n",
                Green.paint("  2.0000e0"),
                Green.paint(" 8.1650e-1"),
                Cyan.paint("  1.0000e0"),
                Red.paint("  3.0000e0"),
                Yellow.paint("[0.2, 0.2]")
            )
        );
    }

    #[test]
    fn prints_statistics_from_all_runs() {
        let sampler = Sampler::new(10, 10, SamplerMode::FitnessSearch, Objective::Single);

        sampler.sample_fitness_single(&2.0, &vec![0.0, 0.1]);
        sampler.sample_fitness_single(&1.0, &vec![0.1, 0.1]);
        sampler.sample_fitness_single(&3.0, &vec![0.2, 0.1]);
        sampler.save_run();
        sampler.end_run();
        sampler.sample_fitness_single(&2.1, &vec![1.0, 0.1]);
        sampler.sample_fitness_single(&0.3, &vec![1.1, 0.1]);
        sampler.sample_fitness_single(&3.2, &vec![1.2, 0.1]);
        sampler.save_run();
        sampler.end_run();
        let mut output = Vec::new();
        sampler.print_statistics(&mut output);

        let output = String::from_utf8(output).expect("Not UTF-8");
        assert_eq!(
            output,
            format!(
                "Number of runs: 2\n\
                 Average {} Standard deviation {}\n\
                 Min: {}. Max: {}\n",
                Green.paint(" 6.5000e-1"),
                Green.paint(" 3.5000e-1"),
                Cyan.paint(" 3.0000e-1"),
                Red.paint("  1.0000e0"),
            ),
        );
    }
}
