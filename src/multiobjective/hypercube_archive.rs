use multiobjective::domination::find_non_dominated;
use operators::selection::roulette_wheel;
use rand::{weak_rng, Rng};
use solution::Solution;
use std::collections::{HashMap, HashSet};
use std::f64::INFINITY;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Debug, PartialEq, Clone)]
struct Hypercube<M>
where
    M: Solution<Vec<f64>> + Eq + Hash,
{
    set: HashSet<M>,
    // TODO: Consider splitting Solution trait into
    // Solution and Position to get rid of this useless field
    position: Vec<f64>,
    fitness: f64,
}

impl<M: Solution<Vec<f64>> + Eq + Hash + Clone> Hypercube<M> {
    fn new(set: HashSet<M>) -> Hypercube<M> {
        let fitness = Hypercube::calculate_fitness(&set);
        Hypercube {
            set,
            fitness,
            position: vec![],
        }
    }

    fn calculate_fitness(set: &HashSet<M>) -> f64 {
        10.0 / set.len() as f64
    }

    fn insert(&mut self, value: M) {
        self.set.insert(value);
        self.fitness = Hypercube::calculate_fitness(&self.set);
    }

    fn remove(&mut self, value: &M) {
        self.set.remove(value);
        self.fitness = Hypercube::calculate_fitness(&self.set);
    }

    fn random(&self) -> &M {
        let mut rng = weak_rng();
        let hypercube_vec: Vec<&M> = self.set.iter().collect();
        rng.choose(&hypercube_vec).unwrap().clone()
    }
}

impl<'a, M: Solution<Vec<f64>> + Eq + Hash> Solution<f64> for &'a Hypercube<M> {
    fn fitness(&self) -> &f64 {
        &self.fitness
    }

    fn position(&self) -> &Vec<f64> {
        &self.position
    }
}

pub struct Archive<M>
where
    M: Solution<Vec<f64>> + Eq + Hash,
{
    hypercube_map: HashMap<Vec<usize>, Hypercube<M>>,
    population_size: usize,
    divisions: usize,
}

impl<M> Archive<M>
where
    M: Solution<Vec<f64>> + Clone + Debug + Eq + Hash,
{
    pub fn new(population_size: usize, divisions: usize) -> Archive<M> {
        Archive {
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

    fn update_hypercube(&mut self, population: Vec<M>) {
        let objectives = population[0].fitness().len();
        let mut min = vec![INFINITY; objectives];
        let mut max = vec![-INFINITY; objectives];
        for solution in &population {
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
        for solution in population.into_iter() {
            let hyper_indices: Vec<usize> = solution
                .fitness()
                .iter()
                .enumerate()
                .map(|(f_i, &f)| self.hypercube_index(min[f_i], max[f_i], f))
                .collect();
            let hypercube = self
                .hypercube_map
                .entry(hyper_indices)
                .or_insert(Hypercube::new(HashSet::new()));
            hypercube.insert(solution);
        }
    }

    fn prune_population(&mut self) {
        while self.get_population().len() > self.population_size {
            let index = {
                let (index, hypercube) = self
                    .hypercube_map
                    .iter_mut()
                    .max_by(|(_, ref h1), (_, ref h2)| h1.set.len().cmp(&h2.set.len()))
                    .unwrap();
                let selected = { hypercube.random().clone() };
                hypercube.remove(&selected);
                index.clone()
            };

            if self.hypercube_map.get(&index).unwrap().set.len() == 0 {
                self.hypercube_map.remove(&index);
            }
        }
    }

    pub fn update(&mut self, population: &[M]) {
        let mut super_population = population.to_vec();
        super_population.append(&mut self.get_population());
        let non_dominated_population: Vec<M> = find_non_dominated(&super_population)
            .iter()
            .map(|p_i| super_population[*p_i].clone())
            .collect();
        self.update_hypercube(non_dominated_population);
        self.prune_population();
    }

    pub fn select_leader(&self) -> &M {
        let hypercubes: Vec<&Hypercube<M>> = self.hypercube_map.values().collect();
        let (_, hypercube) = roulette_wheel(&hypercubes[..]);
        hypercube.random()
    }
    pub fn get_population(&self) -> Vec<M> {
        self.hypercube_map
            .values()
            .flat_map(|hypercube| hypercube.set.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solution::MultiTestSolution;

    fn hashmap_value(
        value: Vec<usize>,
        population: &Vec<MultiTestSolution>,
    ) -> Hypercube<MultiTestSolution> {
        Hypercube::new(value.iter().map(|&v| population[v].clone()).collect())
    }

    #[test]
    fn initializes_correctly() {
        let mut archive: Archive<MultiTestSolution> = Archive::new(7, 5);
        let population = vec![
            MultiTestSolution {
                fitness: vec![0.0, 5.0],
                position: vec![1.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![1.5, 3.9],
                position: vec![2.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![1.8, 3.5],
                position: vec![3.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![2.2, 2.8],
                position: vec![4.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![2.8, 2.2],
                position: vec![5.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![3.5, 1.5],
                position: vec![6.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![5.0, 0.0],
                position: vec![7.0, 2.0],
            },
        ];

        archive.update(&population);

        let mut map = HashMap::new();

        map.insert(vec![0, 4], hashmap_value(vec![0], &population));
        map.insert(vec![1, 3], hashmap_value(vec![1, 2], &population));
        map.insert(vec![2, 2], hashmap_value(vec![3, 4], &population));
        map.insert(vec![3, 1], hashmap_value(vec![5], &population));
        map.insert(vec![4, 0], hashmap_value(vec![6], &population));
        assert_eq!(archive.hypercube_map, map);
    }

    #[test]
    fn updates_correctly() {
        let mut archive: Archive<MultiTestSolution> = Archive::new(7, 5);
        let population = vec![
            MultiTestSolution {
                fitness: vec![0.0, 5.0],
                position: vec![1.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![1.5, 3.9],
                position: vec![2.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![1.8, 3.5],
                position: vec![3.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![2.2, 2.8],
                position: vec![4.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![2.8, 2.2],
                position: vec![5.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![3.5, 1.5],
                position: vec![6.0, 2.0],
            },
            MultiTestSolution {
                fitness: vec![5.0, 0.0],
                position: vec![7.0, 2.0],
            },
        ];

        archive.update(&population[..3]);
        archive.update(&population[3..]);

        let mut map = HashMap::new();

        map.insert(vec![0, 4], hashmap_value(vec![0], &population));
        map.insert(vec![1, 3], hashmap_value(vec![1, 2], &population));
        map.insert(vec![2, 2], hashmap_value(vec![3, 4], &population));
        map.insert(vec![3, 1], hashmap_value(vec![5], &population));
        map.insert(vec![4, 0], hashmap_value(vec![6], &population));
        assert_eq!(archive.hypercube_map, map);
    }
}
