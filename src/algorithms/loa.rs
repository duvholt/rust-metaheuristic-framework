use fitness_evaluation::FitnessEvaluator;
use position::random_position;
use rand::{seq, thread_rng, Rng};
use std::collections::HashSet;
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
    fitness: f64,
    sex: Sex,
}

impl Lion {
    fn key(&self) -> u64 {
        unsafe { mem::transmute(self.fitness) }
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

#[derive(Clone)]
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
            Lion {
                position,
                fitness,
                sex: Sex::None,
            }
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
            Lion {
                position: vec![1.0, 1.0],
                fitness: 1.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![2.0, 2.0],
                fitness: 2.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![3.0, 3.0],
                fitness: 3.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![4.0, 4.0],
                fitness: 4.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![5.0, 5.0],
                fitness: 5.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![6.0, 6.0],
                fitness: 6.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![7.0, 7.0],
                fitness: 7.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![8.0, 8.0],
                fitness: 8.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![9.0, 9.0],
                fitness: 9.0,
                sex: Sex::None,
            },
            Lion {
                position: vec![10.0, 10.0],
                fitness: 10.0,
                sex: Sex::None,
            },
        ]
    }

    fn create_rng() -> StdRng {
        let seed: &[_] = &[1, 2, 3, 4];
        SeedableRng::from_seed(seed)
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
            Lion {
                position: vec![1.0, 1.0],
                fitness: 1.0,
                sex: Sex::Male,
            },
            Lion {
                position: vec![2.0, 2.0],
                fitness: 2.0,
                sex: Sex::Female,
            },
            Lion {
                position: vec![3.0, 3.0],
                fitness: 3.0,
                sex: Sex::Male,
            },
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
            Lion {
                position: vec![1.0, 1.0],
                fitness: 1.0,
                sex: Sex::Male,
            },
            Lion {
                position: vec![3.0, 3.0],
                fitness: 3.0,
                sex: Sex::Male,
            },
        ];
        let pride = Pride {
            population: HashSet::from_iter(&population),
        };
        let mut rng = create_rng();

        let female = pride.random_female(&mut rng);

        assert_eq!(female, None);
    }
}
