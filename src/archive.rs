use solution::MultiSolution;
use std::collections::{HashMap, HashSet};
use std::f64::INFINITY;
use domination::find_non_dominated;

pub struct Archive<M>
where
    M: MultiSolution,
{
    population: Vec<M>,
    hypercube_map: HashMap<Vec<usize>, HashSet<usize>>,
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
            population_size,
            divisions,
        }
    }

    fn hypercube_index(&self, min: f64, max: f64, fitness: f64) -> usize {
        let hypercube_width = (max) / self.divisions as f64;
        let index = ((fitness - min) / hypercube_width) as usize;
        if index == self.divisions {
            // Cheat a bit by making the last hypercube include boundary
            index - 1
        } else {
            index
        }
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
            let hypercube = self.hypercube_map
                .entry(hyper_indices)
                .or_insert(HashSet::new());
            hypercube.insert(s_i);
        }
    }

    pub fn update(&mut self, population: &Vec<M>) {
        let non_dominated = find_non_dominated(&population);
        let nd_population = non_dominated
            .iter()
            .map(|p_i| population[*p_i].clone())
            .collect();
        self.population = nd_population;
        self.update_hypercube();
    }

    pub fn select_leader(&self) -> &M {
        self.population.first().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq)]
    struct TestSolution {
        fitness: Vec<f64>,
    }

    impl MultiSolution for TestSolution {
        fn position(&self) -> &Vec<f64> {
            &self.fitness
        }

        fn fitness(&self) -> &Vec<f64> {
            &self.fitness
        }
    }

    fn find_archive_index(archive: &Archive<TestSolution>, solution: &TestSolution) -> usize {
        archive
            .population
            .iter()
            .position(|ref s| s == &solution)
            .unwrap()
    }

    fn hashmap_value(value: Vec<usize>, indices: &Vec<usize>) -> HashSet<usize> {
        value.iter().map(|&v| indices[v]).collect()
    }

    #[test]
    fn initializes_correctly() {
        let mut archive: Archive<TestSolution> = Archive::new(7, 5);
        let population = vec![
            TestSolution {
                fitness: vec![0.0, 5.0],
            },
            TestSolution {
                fitness: vec![1.5, 3.9],
            },
            TestSolution {
                fitness: vec![1.8, 3.5],
            },
            TestSolution {
                fitness: vec![2.2, 2.8],
            },
            TestSolution {
                fitness: vec![2.8, 2.2],
            },
            TestSolution {
                fitness: vec![3.5, 1.5],
            },
            TestSolution {
                fitness: vec![5.0, 0.0],
            },
        ];

        archive.update(&population);

        let mut map = HashMap::new();
        let indices: Vec<_> = population
            .iter()
            .map(|ref solution| find_archive_index(&archive, &solution))
            .collect();
        map.insert(vec![0, 4], hashmap_value(vec![0], &indices));
        map.insert(vec![1, 3], hashmap_value(vec![1, 2], &indices));
        map.insert(vec![2, 2], hashmap_value(vec![3, 4], &indices));
        map.insert(vec![3, 1], hashmap_value(vec![5], &indices));
        map.insert(vec![4, 0], hashmap_value(vec![6], &indices));
        assert_eq!(archive.hypercube_map, map);
    }
}
