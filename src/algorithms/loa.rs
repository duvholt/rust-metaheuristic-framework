use fitness_evaluation::FitnessEvaluator;
use position::{euclidean_distance, perpendicular_position, random_position};
use rand::{seq, thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::hash;
use std::iter::FromIterator;
use std::mem;

struct Config {
    iterations: i64,
    population: usize,
    upper_bound: f64,
    lower_bound: f64,
    dimension: usize,
    prides: usize,
    nomad_percent: f64,
    roaming_percent: f64,
    mutation_probability: f64,
    sex_rate: f64,
    mating_probability: f64,
    immigate_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum Sex {
    Male,
    Female,
    None,
}

#[derive(Debug, Clone)]
struct Lion {
    position: Vec<f64>,
    best_position: Vec<f64>,
    fitness: f64,
    prev_fitness: f64,
    sex: Sex,
}

impl Lion {
    fn new(position: Vec<f64>, fitness: f64) -> Lion {
        let best_position = position.clone();
        Lion {
            position,
            best_position,
            fitness,
            prev_fitness: fitness,
            sex: Sex::None,
        }
    }

    fn update_position(&mut self, position: Vec<f64>, fitness: f64) {
        if self.fitness > fitness {
            self.best_position = position.clone();
        }
        self.prev_fitness = self.fitness;
        self.position = position;
        self.fitness = fitness;
    }

    fn key(&self) -> u64 {
        unsafe { mem::transmute(self.fitness) }
    }

    fn diff_position(&self, other: &Vec<f64>) -> Vec<f64> {
        self.position
            .iter()
            .zip(other.iter())
            .map(|(p1, p2)| p2 - p1)
            .collect()
    }

    fn has_improved(&self) -> bool {
        self.fitness < self.prev_fitness
    }
}

impl hash::Hash for Lion {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.key().hash(state)
    }
}

impl PartialEq for Lion {
    fn eq(&self, other: &Lion) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Lion {}

#[derive(Clone, Debug)]
struct Pride<'a> {
    population: HashSet<&'a Lion>,
}

impl<'a> Pride<'a> {
    fn random_female(&self, mut rng: impl Rng) -> Option<&'a Lion> {
        let females = self.population
            .iter()
            .cloned()
            .filter(|lion| lion.sex == Sex::Female);
        let sample = seq::sample_iter(&mut rng, females, 1).ok()?;
        Some(sample.first()?)
    }
}

struct Nomad<'a> {
    population: HashSet<&'a Lion>,
}

fn random_population(config: &Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<Lion> {
    (0..config.population)
        .map(|_| {
            let position =
                random_position(config.lower_bound, config.upper_bound, config.dimension);
            let fitness = fitness_evaluator.calculate_fitness(&position);
            Lion::new(position, fitness)
        })
        .collect()
}

fn find_sex(r: f64, sex_rate: f64, nomad: bool) -> Sex {
    let female = if nomad { r > sex_rate } else { r < sex_rate };
    if female {
        Sex::Female
    } else {
        Sex::Male
    }
}

fn partition_lions<'a>(
    config: &Config,
    population: &'a mut Vec<Lion>,
    rng: &mut impl Rng,
) -> (Nomad<'a>, Vec<Pride<'a>>) {
    let last_nomad_index = (population.len() as f64 * config.nomad_percent) as usize;
    for (i, lion) in population.iter_mut().enumerate() {
        let r: f64 = rng.gen();
        lion.sex = find_sex(r, config.sex_rate, i > last_nomad_index);
    }
    let nomad = Nomad {
        population: HashSet::from_iter(population[..last_nomad_index].iter()),
    };
    let mut prides = vec![
        Pride {
            population: HashSet::new()
        };
        config.prides
    ];
    for lion in population[last_nomad_index..].iter() {
        let pride_index = rng.gen_range(0, config.prides);
        prides[pride_index].population.insert(lion);
    }
    (nomad, prides)
}

fn find_hunters<'a>(prides: &'a Vec<Pride>, mut rng: impl Rng) -> Vec<&'a Lion> {
    prides
        .iter()
        .map(|pride| pride.random_female(&mut rng))
        .filter(|female| female.is_some())
        .map(|female| female.unwrap())
        .collect()
}

fn calculate_prey_position(hunters: &Vec<&mut Lion>, dimensions: usize) -> Vec<f64> {
    hunters
        .iter()
        .fold(vec![0.0; dimensions], |prey, hunter| {
            prey.iter()
                .zip(hunter.position.iter())
                .map(|(prey_i, hunter_i)| prey_i + hunter_i)
                .collect()
        })
        .iter()
        .map(|prey_i| prey_i / hunters.len() as f64)
        .collect()
}

fn group_hunters(hunters: Vec<&mut Lion>, mut rng: impl Rng) -> Vec<Vec<&mut Lion>> {
    let mut groups = vec![vec![], vec![], vec![]];
    for mut hunter in hunters {
        let index = rng.gen_range(0, 3);
        groups[index].push(hunter);
    }
    groups
}

fn find_center_group(groups: &Vec<Vec<&mut Lion>>) -> usize {
    let cumulative_fitness = groups
        .iter()
        .map(|group| group.iter().map(|lion| lion.fitness).sum());

    // Find index of largest cumulative fitness
    let (index, _) = cumulative_fitness
        .enumerate()
        .max_by(|(_, f1): &(usize, f64), (_, f2): &(usize, f64)| {
            f1.partial_cmp(&f2).unwrap_or(Ordering::Equal)
        })
        .unwrap();
    index
}

fn hunting_position_wing(hunter: &Vec<f64>, prey: &Vec<f64>, mut rng: impl Rng) -> Vec<f64> {
    hunter
        .iter()
        .zip(prey)
        .map(|(hunter_i, prey_i)| {
            let val = 2.0 * *prey_i - *hunter_i;
            if val < *prey_i {
                rng.gen_range(val, *prey_i)
            } else {
                rng.gen_range(*prey_i, val)
            }
        })
        .collect()
}

fn hunting_position_center(hunter: &Vec<f64>, prey: &Vec<f64>, mut rng: impl Rng) -> Vec<f64> {
    hunter
        .iter()
        .zip(prey)
        .map(|(hunter_i, prey_i)| {
            if *hunter_i < *prey_i {
                rng.gen_range(*hunter_i, *prey_i)
            } else {
                rng.gen_range(*prey_i, *hunter_i)
            }
        })
        .collect()
}

fn update_prey(hunter: &Vec<f64>, prey: &Vec<f64>, pi: f64, mut rng: impl Rng) -> Vec<f64> {
    prey.iter()
        .zip(hunter)
        .map(|(prey_i, hunter_i)| prey_i + rng.gen::<f64>() * pi * (prey_i - hunter_i))
        .collect()
}

fn hunt(
    hunters: Vec<&mut Lion>,
    dimensions: usize,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<f64>,
) {
    let mut prey = calculate_prey_position(&hunters, dimensions);
    let mut groups = group_hunters(hunters, &mut rng);
    let center_group_index = find_center_group(&groups);
    for (i, group) in groups.iter_mut().enumerate() {
        for mut hunter in group {
            let position = if i == center_group_index {
                hunting_position_center(&hunter.position, &prey, &mut rng)
            } else {
                hunting_position_wing(&hunter.position, &prey, &mut rng)
            };
            let fitness = fitness_evaluator.calculate_fitness(&position);
            if fitness < hunter.fitness {
                hunter.update_position(position, fitness);
                prey = update_prey(&hunter.position, &prey, hunter.fitness - fitness, &mut rng);
            }
        }
    }
}

fn calculate_tournament_size(lions: &[&mut Lion]) -> usize {
    lions.iter().map(|lion| lion.has_improved() as usize).sum()
}

fn move_towards_safe_place(lion: &mut Lion, selected_lion: &Lion, mut rng: impl Rng) {
    let distance = euclidean_distance(&lion.position, &selected_lion.position);
    let r1 = lion.diff_position(&selected_lion.position);
    let r2 = perpendicular_position(&r1, &mut rng);
    let r: f64 = rng.gen();
    let u = rng.gen_range(-1.0, 1.0);
    let theta = rng.gen_range(-PI / 6.0, PI / 6.0);
    for i in 0..lion.position.len() {
        let r1_i = r1[i];
        let r2_i = r2[i];
        lion.position[i] += 2.0 * distance * r * r1_i + u * theta.tan() * distance * r2_i;
    }
}

fn roam_pride(
    lion: &mut Lion,
    pride: &Pride,
    roaming_percent: f64,
    fitness_evaluator: &FitnessEvaluator<f64>,
    mut rng: impl Rng,
) {
    let roaming_count = (roaming_percent * pride.population.len() as f64) as usize;
    let selected = seq::sample_iter(&mut rng, &pride.population, roaming_count).unwrap();
    let roam_positions: Vec<_> = selected
        .into_iter()
        .map(|lion| &lion.best_position)
        .collect();
    for roam_position in roam_positions {
        let distance = euclidean_distance(&lion.position, &roam_position);
        let x = rng.gen_range(0.0, 2.0 * distance);
        let theta = rng.gen_range(-PI / 6.0, PI / 6.0);
        let position: Vec<_> = lion.position
            .iter()
            .map(|p_i| p_i + x * theta.tan())
            .collect();
        let fitness = fitness_evaluator.calculate_fitness(&position);
        lion.update_position(position, fitness);
    }
}

fn min_value(val1: f64, val2: f64) -> f64 {
    if val1 > val2 {
        val1
    } else {
        val2
    }
}

fn roam_nomad(
    nomad: &mut Lion,
    best: &Lion,
    lower_bound: f64,
    upper_bound: f64,
    fitness_evaluator: &FitnessEvaluator<f64>,
    mut rng: impl Rng,
) {
    let pr = 0.1 + min_value(0.5, (nomad.fitness - best.fitness) / best.fitness);
    let position: Vec<_> = nomad
        .position
        .iter()
        .map(|p_i| {
            let r: f64 = rng.gen();
            if r > pr {
                *p_i
            } else {
                rng.gen_range(lower_bound, upper_bound)
            }
        })
        .collect();
    let fitness = fitness_evaluator.calculate_fitness(&position);
    nomad.update_position(position, fitness);
}

fn mutate_random(position: &Vec<f64>, mutation_probability: f64, lower_bound: f64, upper_bound: f64, mut rng: impl Rng) -> Vec<f64> {
    position.iter().map(|pos_i| {
        let r: f64 = rng.gen();
        if r < mutation_probability {
            rng.gen_range(lower_bound, upper_bound)
        } else {
            *pos_i
        }
    }).collect()
}

// TODO: Replace with generic sort

fn sort_lions(population: &mut Vec<&mut Lion>) {
    population
        .sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
}

fn defense_resident_male<'a>(old_males: Vec<&'a mut Lion>, mut new_males: Vec<&'a mut Lion>) -> (Vec<&'a mut Lion>, Vec<&'a mut Lion>) {
    let old_males_size = old_males.len();
    let mut males = old_males;
    males.append(&mut new_males);
    sort_lions(&mut males);
    let nomads = males.split_off(old_males_size);
    (males, nomads)
}

fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) {
    let population = random_population(&config, &fitness_evaluator);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, StdRng};
    use testing::utils::{create_evaluator, create_sampler};

    fn create_config() -> Config {
        Config {
            iterations: 100,
            population: 10,
            upper_bound: 1.0,
            lower_bound: -1.0,
            dimension: 2,
            prides: 4,
            nomad_percent: 0.2,
            roaming_percent: 0.2,
            mutation_probability: 0.2,
            sex_rate: 0.8,
            mating_probability: 0.3,
            immigate_rate: 0.4,
        }
    }

    fn create_population() -> Vec<Lion> {
        vec![
            Lion::new(vec![1.0, 1.0], 1.0),
            Lion::new(vec![2.0, 2.0], 2.0),
            Lion::new(vec![3.0, 3.0], 3.0),
            Lion::new(vec![4.0, 4.0], 4.0),
            Lion::new(vec![5.0, 5.0], 5.0),
            Lion::new(vec![6.0, 6.0], 6.0),
            Lion::new(vec![7.0, 7.0], 7.0),
            Lion::new(vec![8.0, 8.0], 8.0),
            Lion::new(vec![9.0, 9.0], 9.0),
            Lion::new(vec![10.0, 10.0], 10.0),
        ]
    }

    fn create_rng() -> StdRng {
        let seed: &[_] = &[1, 2, 3, 4];
        SeedableRng::from_seed(seed)
    }

    fn create_lion_with_sex(position: Vec<f64>, fitness: f64, sex: Sex) -> Lion {
        let mut lion = Lion::new(position, fitness);
        lion.sex = sex;
        lion
    }

    #[test]
    fn creates_population_with_correct_size() {
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);
        let config = create_config();

        let population = random_population(&config, &fitness_evaluator);

        assert_eq!(population.len(), config.population);
    }

    #[test]
    fn partitions_into_nomad_and_prides() {
        let config = create_config();
        let mut population = create_population();
        let mut rng = create_rng();

        let (nomad, prides) = partition_lions(&config, &mut population, &mut rng);

        assert_eq!(nomad.population.len(), 2);
        assert_eq!(prides.len(), 4);
        let prides_sum: usize = prides.iter().map(|p| p.population.len()).sum();
        assert_eq!(prides_sum, 8);
        // Check that all lions have been assigned a sex
        for lion in nomad.population.iter() {
            assert!(Sex::None != lion.sex);
        }
        for lion in prides.iter().flat_map(|p| &p.population) {
            assert!(Sex::None != lion.sex);
        }
    }

    #[test]
    fn assigns_nomad_sex_female() {
        let r = 0.9;
        let sex_rate = 0.8;
        let nomad = true;

        let sex = find_sex(r, sex_rate, nomad);

        assert_eq!(sex, Sex::Female);
    }

    #[test]
    fn assigns_nomad_sex_male() {
        let r = 0.5;
        let sex_rate = 0.8;
        let nomad = true;

        let sex = find_sex(r, sex_rate, nomad);

        assert_eq!(sex, Sex::Male);
    }

    #[test]
    fn assigns_pride_sex_female() {
        let r = 0.6;
        let sex_rate = 0.8;
        let nomad = false;

        let sex = find_sex(r, sex_rate, nomad);

        assert_eq!(sex, Sex::Female);
    }

    #[test]
    fn assigns_pride_sex_male() {
        let r = 0.85;
        let sex_rate = 0.8;
        let nomad = false;

        let sex = find_sex(r, sex_rate, nomad);

        assert_eq!(sex, Sex::Male);
    }

    #[test]
    fn finds_female_in_pride() {
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0], 1.0, Sex::Male),
            create_lion_with_sex(vec![2.0, 2.0], 2.0, Sex::Female),
            create_lion_with_sex(vec![3.0, 3.0], 3.0, Sex::Male),
        ];
        let pride = Pride {
            population: HashSet::from_iter(&population),
        };
        let mut rng = create_rng();

        let female = pride.random_female(&mut rng);

        assert_eq!(female, Some(&population[1]));
    }

    #[test]
    fn does_not_find_female_in_pride() {
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0], 1.0, Sex::Male),
            create_lion_with_sex(vec![3.0, 3.0], 3.0, Sex::Male),
        ];
        let pride = Pride {
            population: HashSet::from_iter(&population),
        };
        let mut rng = create_rng();

        let female = pride.random_female(&mut rng);

        assert_eq!(female, None);
    }

    #[test]
    fn finds_one_hunter_from_all_prides() {
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0], 1.0, Sex::Male),
            create_lion_with_sex(vec![2.0, 2.0], 2.0, Sex::Female),
            create_lion_with_sex(vec![3.0, 3.0], 3.0, Sex::Male),
            create_lion_with_sex(vec![4.0, 4.0], 4.0, Sex::Female),
            create_lion_with_sex(vec![5.0, 5.0], 5.0, Sex::Male),
            create_lion_with_sex(vec![6.0, 6.0], 6.0, Sex::Female),
        ];
        let prides = vec![
            Pride {
                population: HashSet::from_iter(&population[..2]),
            },
            Pride {
                population: HashSet::from_iter(&population[2..4]),
            },
            Pride {
                population: HashSet::from_iter(&population[4..6]),
            },
        ];
        let rng = create_rng();

        let hunters = find_hunters(&prides, rng);

        assert_eq!(
            hunters,
            vec![&population[1], &population[3], &population[5]]
        );
    }

    #[test]
    fn calculates_prey_position_correctly() {
        let mut population = vec![
            create_lion_with_sex(vec![2.0, 3.0], 3.0, Sex::Female),
            create_lion_with_sex(vec![7.0, 2.0], 3.0, Sex::Female),
        ];
        let hunters = population.iter_mut().map(|l| l).collect();

        let prey_position = calculate_prey_position(&hunters, 2);

        assert_eq!(prey_position, vec![9.0 / 2.0, 5.0 / 2.0]);
    }

    #[test]
    fn partitions_hunters_randomly() {
        let mut population = vec![
            create_lion_with_sex(vec![2.0, 3.0], 3.0, Sex::Female),
            create_lion_with_sex(vec![7.0, 1.0], 2.0, Sex::Female),
            create_lion_with_sex(vec![3.0, 6.0], 7.0, Sex::Female),
            create_lion_with_sex(vec![2.0, 3.0], 4.0, Sex::Female),
        ];
        let hunters: Vec<&mut Lion> = population.iter_mut().map(|l| l).collect();
        let rng = create_rng();

        let groups = group_hunters(hunters, rng);

        let lions_in_groups: Vec<_> = groups.iter().flat_map(|g| g).collect();
        assert_eq!(groups.len(), 3);
        assert_eq!(lions_in_groups.len(), 4);
    }

    #[test]
    fn selects_group_with_highest_fitness() {
        let mut population = vec![
            create_lion_with_sex(vec![3.0, 6.0], 6.0, Sex::Female),
            create_lion_with_sex(vec![2.0, 3.0], 3.0, Sex::Female),
            create_lion_with_sex(vec![2.0, 3.0], 5.0, Sex::Female),
            create_lion_with_sex(vec![7.0, 1.0], 2.0, Sex::Female),
        ];
        let (first, last) = population.split_at_mut(2);
        let (l1, l2) = first.split_at_mut(1);
        let (l3, l4) = last.split_at_mut(1);
        let groups = vec![
            vec![&mut l1[0]],
            vec![&mut l2[0], &mut l3[0]],
            vec![&mut l4[0]],
        ];

        let center_index = find_center_group(&groups);

        assert_eq!(center_index, 1);
    }

    #[test]
    fn calculates_hunting_position_wing_correctly() {
        let hunter = vec![0.4, 0.1, 0.5];
        let prey = vec![0.2, 0.5, 0.6];
        let rng = create_rng();

        let position = hunting_position_wing(&hunter, &prey, rng);

        assert_eq!(position.len(), 3);
    }

    #[test]
    fn calculates_hunting_position_center_correctly() {
        let hunter = vec![0.4, 0.1, 0.5];
        let prey = vec![0.2, 0.5, 0.6];
        let rng = create_rng();

        let position = hunting_position_center(&hunter, &prey, rng);

        assert_eq!(position.len(), 3);
    }

    #[test]
    fn calculates_new_prey_position() {
        let hunter = vec![0.2, 0.3];
        let prey = vec![0.1, 0.4];
        let rng = create_rng();

        let new_prey = update_prey(&hunter, &prey, 5.0, rng);

        assert_eq!(new_prey.len(), 2);
    }

    #[test]
    fn updates_lion_position() {
        let mut lion = Lion::new(vec![0.1, 0.2], 0.3);

        lion.update_position(vec![0.4, 0.5], 0.6);

        assert_eq!(lion.position, vec![0.4, 0.5]);
        assert_eq!(lion.fitness, 0.6);
        assert_eq!(lion.sex, Sex::None);
        assert_eq!(lion.best_position, vec![0.1, 0.2]);
    }

    #[test]
    fn keeps_track_of_previous_fitness() {
        let mut lion = Lion::new(vec![0.1, 0.2], 0.3);

        lion.update_position(vec![0.4, 0.5], 0.6);

        assert_eq!(lion.prev_fitness, 0.3);
    }

    #[test]
    fn finds_diff_position() {
        let lion = Lion::new(vec![0.1, 0.2, 0.3], 4.0);

        let diff = lion.diff_position(&vec![0.05, 0.2, 0.4]);

        assert_eq!(diff, vec![-0.05, 0.0, 0.10000000000000003]);
    }

    #[test]
    fn moves_lion() {
        let mut lion = Lion::new(vec![2.0, 3.0, 4.0], 2.0);
        let selected_lion = Lion::new(vec![1.0, 6.0, 2.0], 1.0);
        let mut rng = create_rng();

        move_towards_safe_place(&mut lion, &selected_lion, rng);

        // More or less hoping that this is the correct result
        assert_eq!(
            lion.position,
            vec![-0.6874672751682849, 14.561451594829403, -2.697461529705847]
        );
    }

    #[test]
    fn counts_improved_lions() {
        let mut lion1 = Lion::new(vec![0.1, 0.2, 0.31], 0.31);
        let mut lion2 = Lion::new(vec![0.1, 0.2, 0.32], 0.32);
        let mut lion3 = Lion::new(vec![0.1, 0.2, 0.33], 0.33);
        lion1.update_position(vec![0.1, 0.2, 0.21], 0.21);
        lion3.update_position(vec![0.1, 0.2, 0.23], 0.23);

        let tournament_size = calculate_tournament_size(&vec![&mut lion1, &mut lion2, &mut lion3]);

        assert_eq!(tournament_size, 2);
    }

    #[test]
    fn moves_lion_with_roam_nomad() {
        let mut lion = create_lion_with_sex(vec![0.2, 0.1, 0.3], 0.3, Sex::Male);
        let best = create_lion_with_sex(vec![0.2, 0.1, 0.0], 0.1, Sex::Male);
        let mut rng = create_rng();
        let sampler = create_sampler();
        let mut fitness_evaluator = create_evaluator(&sampler);

        let original_position = lion.position.clone();
        roam_nomad(&mut lion, &best, -10.0, 10.0, &fitness_evaluator, &mut rng);

        assert!(lion.position != original_position);
    }

    #[test]
    fn moves_lion_with_roam_pride() {
        let mut lion = create_lion_with_sex(vec![0.2, 0.1, 0.3], 0.3, Sex::Male);
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0, 0.3], 1.1, Sex::Male),
            create_lion_with_sex(vec![2.0, 2.0, 0.2], 2.2, Sex::Female),
            create_lion_with_sex(vec![3.2, 0.0, 1.1], 1.3, Sex::Male),
            create_lion_with_sex(vec![4.0, 1.0, 0.1], 2.4, Sex::Male),
            create_lion_with_sex(vec![3.0, 5.0, 2.1], 1.5, Sex::Male),
        ];
        let pride = Pride {
            population: HashSet::from_iter(&population),
        };
        let mut rng = create_rng();
        let sampler = create_sampler();
        let mut fitness_evaluator = create_evaluator(&sampler);

        let original_position = lion.position.clone();
        roam_pride(&mut lion, &pride, 0.4, &fitness_evaluator, &mut rng);

        assert!(lion.position != original_position);
    }

    #[test]
    fn mutates_randomly() {
        let position = vec![0.0, 0.1, 0.2, 0.3];
        let rng = create_rng();

        let new_position = mutate_random(&position, 0.3, -10.0, 10.0, rng);

        assert!(position != new_position);
    }

    #[test]
    fn defends_against_resident_males() {
        let mut population = vec![
            create_lion_with_sex(vec![0.5, 0.5], 0.5, Sex::Male),
            create_lion_with_sex(vec![0.3, 0.3], 0.3, Sex::Male),
            create_lion_with_sex(vec![0.2, 0.2], 0.2, Sex::Male),
            create_lion_with_sex(vec![0.4, 0.4], 0.4, Sex::Male),
            create_lion_with_sex(vec![0.6, 0.6], 0.6, Sex::Male),
            create_lion_with_sex(vec![0.1, 0.1], 0.1, Sex::Male),
        ];
        let (old_males, new_males) = population.split_at_mut(3);
        let old_males: Vec<&mut Lion> = old_males.iter_mut().map(|lion| lion).collect();
        let new_males: Vec<&mut Lion> = new_males.iter_mut().map(|lion| lion).collect();

        let (pride_males, nomads) = defense_resident_male(old_males, new_males);

        // Checking fitness only as several mutable referens does not work good in Rust
        let pride_males_fitness: Vec<_> = pride_males.iter().map(|l| l.fitness).collect();
        assert_eq!(pride_males_fitness, vec![0.1, 0.2, 0.3]);
        let nomads_fitness: Vec<_> = nomads.iter().map(|l| l.fitness).collect();
        assert_eq!(nomads_fitness, vec![0.4, 0.5, 0.6]);
    }
}
