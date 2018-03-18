use solution::MultiSolution;
use std::collections::HashMap;
use std::f64::INFINITY;
use domination::find_non_dominated;

pub struct Archive<M>
where
    M: MultiSolution,
{
    population: Vec<M>,
    hypercube_map: HashMap<Vec<usize>, Vec<usize>>,
    hypercube_reverse: HashMap<usize, Vec<usize>>,
    population_size: usize,
    divisions: usize,
}

impl<M> Archive<M>
where
    M: MultiSolution + Clone,
{
    pub fn new(population_size: usize, divisions: usize) -> Archive<M> {
        Archive {
            population: vec![],
            hypercube_map: HashMap::new(),
            hypercube_reverse: HashMap::new(),
            population_size,
            divisions,
        }
    }

    fn hypercube_index(&self, min: f64, max: f64, fitness: f64) -> usize {
        // TODO: Check if multiplying by 1.001 actually works as intended
        let hypercube_width = (max * 1.001) / self.divisions as f64;
        ((fitness - min) / hypercube_width) as usize
    }

    fn update_hypercube(&mut self) {
        let mut min = vec![INFINITY; self.population[0].fitness().len()];
        let mut max = vec![-INFINITY; self.population[0].fitness().len()];
        for solution in &self.population {
            for (f_i, &fitness) in solution.fitness().iter().enumerate() {
                if min[f_i] > fitness {
                    min[f_i] = fitness;
                }
                if max[f_i] < fitness {
                    max[f_i] = fitness;
                }
            }
        }
        self.hypercube_map.clear();
        for (s_i, solution) in self.population.iter().enumerate() {
            let hyper_indices: Vec<usize> = solution
                .fitness()
                .iter()
                .enumerate()
                .map(|(f_i, &f)| self.hypercube_index(min[f_i], max[f_i], f))
                .collect();
            let hypercube = self.hypercube_map.entry(hyper_indices).or_insert(vec![]);
            hypercube.push(s_i);
        }
    }

    pub fn update(&mut self, population: &Vec<M>) {
        let non_dominated = find_non_dominated(&population);
        let nd_population = non_dominated
            .iter()
            .map(|p_i| population[*p_i].clone())
            .collect();
        self.population = nd_population;
    }

    pub fn select_leader(&self) -> &M {
        self.population.first().unwrap()
    }
}
