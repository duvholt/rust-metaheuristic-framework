use itertools::Itertools;
use multiobjective::domination::find_non_dominated;
use multiobjective::non_dominated_sorting::crowding_distance;
use operators::selection::tournament_selection_crowding;
use rand::weak_rng;
use solution::Solution;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

pub struct Archive<M>
where
    M: Solution<Vec<f64>>,
{
    solutions: Vec<M>,
    archive_size: usize,
    crowding_distance: Vec<f64>,
}

impl<M> Archive<M>
where
    M: Solution<Vec<f64>> + Debug + Clone + Hash + Eq,
{
    pub fn new(archive_size: usize) -> Archive<M> {
        Archive {
            solutions: vec![],
            archive_size,
            crowding_distance: vec![],
        }
    }

    pub fn update(&mut self, population: &[M]) {
        self.solutions.append(&mut population.to_vec());
        self.solutions = find_non_dominated(&self.solutions)
            .iter()
            .map(|p_i| self.solutions[*p_i].clone())
            .collect();
        self.crowding_distance = crowding_distance(&self.solutions);
        if self.solutions.len() > self.archive_size {
            self.prune();
        }
    }

    fn prune(&mut self) {
        let distance = self.crowding_distance
            .iter()
            .cloned()
            .enumerate()
            .sorted_by(|(_, distance), (_, distance2)| distance.partial_cmp(&distance2).unwrap());
        let set: HashSet<_> = distance
            .into_iter()
            .take(self.archive_size)
            .map(|(i, _)| self.solutions[i].clone())
            .collect();
        self.solutions.retain(|solution| set.contains(solution));
        self.crowding_distance = crowding_distance(&self.solutions);
    }

    pub fn select_leader(&self) -> &M {
        let best_index =
            tournament_selection_crowding(&self.solutions, 2, weak_rng(), &self.crowding_distance);
        &self.solutions[best_index]
    }

    pub fn get_population(&self) -> Vec<M> {
        self.solutions.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solution::MultiTestSolution;
}
