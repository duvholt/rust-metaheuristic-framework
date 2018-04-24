use clap::{App, Arg, ArgMatches, SubCommand};
use config::CommonConfig;
use crossover::uniform;
use fitness_evaluation::FitnessEvaluator;
use position::{limit_position, perpendicular_position, random_position};
use rand::distributions::{IndependentSample, Normal};
use rand::{seq, thread_rng, Rng};
use selection::tournament_selection;
use solution::{Solution, SolutionJSON};
use std::cmp::max;
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::hash;
use std::mem;

pub fn subcommand(name: &str) -> App<'static, 'static> {
    SubCommand::with_name(name)
        .about("lion optimization algorithm")
        .arg(
            Arg::with_name("prides")
                .long("prides")
                .value_name("prides")
                .help("prides constant")
                .default_value("4")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("nomad_percent")
                .long("nomad_percent")
                .value_name("nomad_percent")
                .help("nomad_percent constant")
                .default_value("0.2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("roaming_percent")
                .long("roaming_percent")
                .value_name("roaming_percent")
                .help("roaming_percent constant")
                .default_value("0.2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("mutation")
                .long("mutation")
                .value_name("mutation")
                .help("mutation constant")
                .default_value("0.2")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("sex_rate")
                .long("sex_rate")
                .value_name("sex_rate")
                .help("sex_rate constant")
                .default_value("0.8")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("mating")
                .long("mating")
                .value_name("mating")
                .help("mating constant")
                .default_value("0.3")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("immigrate_rate")
                .long("immigrate_rate")
                .value_name("immigrate_rate")
                .help("immigrate_rate constant")
                .default_value("0.4")
                .takes_value(true),
        )
}

pub fn run_subcommand(
    common: &CommonConfig,
    function_evaluator: &FitnessEvaluator<f64>,
    sub_m: &ArgMatches,
) -> Vec<SolutionJSON> {
    let prides = value_t_or_exit!(sub_m, "prides", usize);
    let nomad_percent = value_t_or_exit!(sub_m, "nomad_percent", f64);
    let roaming_percent = value_t_or_exit!(sub_m, "roaming_percent", f64);
    let mutation = value_t_or_exit!(sub_m, "mutation", f64);
    let sex_rate = value_t_or_exit!(sub_m, "sex_rate", f64);
    let mating = value_t_or_exit!(sub_m, "mating", f64);
    let immigrate_rate = value_t_or_exit!(sub_m, "immigrate_rate", f64);

    let config = Config {
        verbose: common.verbose,
        upper_bound: common.upper_bound,
        lower_bound: common.lower_bound,
        dimensions: common.dimensions,
        iterations: common.iterations,
        population: common.population,
        prides,
        nomad_percent,
        roaming_percent,
        mutation_probability: mutation,
        sex_rate,
        mating_probability: mating,
        immigate_rate: immigrate_rate,
    };
    println!("Running LOA with {:?}", config);
    run(config, &function_evaluator)
}

#[derive(Debug)]
struct Config {
    verbose: bool,
    iterations: i64,
    population: usize,
    upper_bound: f64,
    lower_bound: f64,
    dimensions: usize,
    prides: usize,
    nomad_percent: f64,
    roaming_percent: f64,
    mutation_probability: f64,
    sex_rate: f64,
    mating_probability: f64,
    immigate_rate: f64,
}

impl Config {
    fn females_in_pride(&self) -> usize {
        ((self.population as f64 * (1.0 - self.nomad_percent) * self.sex_rate) / self.prides as f64)
            as usize
    }

    fn males_in_pride(&self) -> usize {
        ((self.population as f64 * (1.0 - self.nomad_percent) * (1.0 - self.sex_rate))
            / self.prides as f64)
            .round() as usize
    }

    fn pride_size(&self) -> usize {
        (self.population as f64 * (1.0 - self.nomad_percent) / self.prides as f64) as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Sex {
    Male,
    Female,
    None,
}

#[derive(Debug, Clone, PartialEq)]
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

impl Solution<f64> for Lion {
    fn position(&self) -> &Vec<f64> {
        &self.position
    }

    fn fitness(&self) -> &f64 {
        &self.fitness
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

impl Eq for Lion {}

impl PartialOrd for Lion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fitness.partial_cmp(&other.fitness)
    }
}

impl Ord for Lion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Debug)]
struct Pride {
    population: Vec<Lion>,
}

impl Pride {
    fn random_female(&self, mut rng: impl Rng) -> Option<&Lion> {
        let females = self.population
            .iter()
            .filter(|lion| lion.sex == Sex::Female);

        let lion: Vec<&Lion> = seq::sample_iter(&mut rng, females, 1).ok()?;
        Some(lion.first()?)
    }
}

#[derive(Debug)]
struct Nomad {
    population: Vec<Lion>,
}

fn random_population(config: &Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<Lion> {
    (0..config.population)
        .map(|_| {
            let position =
                random_position(config.lower_bound, config.upper_bound, config.dimensions);
            let fitness = fitness_evaluator.calculate_fitness(&position);
            Lion::new(position, fitness)
        })
        .collect()
}

fn partition_lions(config: &Config, mut population: Vec<Lion>) -> (Nomad, Vec<Pride>) {
    let last_nomad_index = (population.len() as f64 * config.nomad_percent) as usize;
    for (i, lion) in population.iter_mut().enumerate() {
        let sex = if i > last_nomad_index {
            if (i - last_nomad_index) % config.pride_size() < config.females_in_pride() {
                Sex::Female
            } else {
                Sex::Male
            }
        } else {
            let females_in_nomad = (config.population as f64 * config.nomad_percent
                * (1.0 - config.sex_rate as f64))
                .round() as usize;
            if i < females_in_nomad {
                Sex::Female
            } else {
                Sex::Male
            }
        };
        lion.sex = sex;
    }
    let pride_lions = population.split_off(last_nomad_index);
    let nomad = Nomad {
        population: population.into_iter().collect(),
    };
    let mut prides = vec![
        Pride {
            population: Vec::new()
        };
        config.prides
    ];
    let pride_size = config.pride_size();
    for (pride_index, mut lions) in pride_lions.chunks(pride_size).into_iter().enumerate() {
        prides[pride_index]
            .population
            .append(&mut lions.into_iter().cloned().collect());
    }
    (nomad, prides)
}

fn find_hunters(prides: &mut Vec<Pride>, mut rng: impl Rng) -> Vec<(usize, Lion)> {
    prides
        .iter_mut()
        .enumerate()
        .filter_map(|(i, pride)| {
            let random_female = pride.random_female(&mut rng)?.clone();
            let index = pride
                .population
                .iter()
                .position(|l| l == &random_female)
                .unwrap();
            Some((i, pride.population.remove(index)))
        })
        .collect()
}

fn calculate_prey_position(hunters: &Vec<&Lion>, dimensions: usize) -> Vec<f64> {
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

fn group_hunters(hunters: Vec<(usize, Lion)>, mut rng: impl Rng) -> Vec<Vec<(usize, Lion)>> {
    let mut groups = vec![vec![], vec![], vec![]];
    for hunter in hunters.into_iter() {
        let index = rng.gen_range(0, 3);
        groups[index].push(hunter);
    }
    groups
}

fn find_center_group(groups: &Vec<Vec<(usize, Lion)>>) -> usize {
    let cumulative_fitness = groups
        .iter()
        .map(|group| group.iter().map(|(_, lion)| lion.fitness).sum());

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
            if val == *prey_i {
                val
            } else if val < *prey_i {
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
            if *hunter_i == *prey_i {
                *hunter_i
            } else if *hunter_i < *prey_i {
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
    hunters: Vec<(usize, Lion)>,
    dimensions: usize,
    lower_bound: f64,
    upper_bound: f64,
    mut rng: impl Rng,
    fitness_evaluator: &FitnessEvaluator<f64>,
) -> Vec<(usize, Lion)> {
    let mut prey =
        calculate_prey_position(&hunters.iter().map(|(_, lion)| lion).collect(), dimensions);
    let mut new_hunters = Vec::with_capacity(hunters.len());
    let groups = group_hunters(hunters, &mut rng);
    let center_group_index = find_center_group(&groups);
    for (i, group) in groups.into_iter().enumerate() {
        for (j, mut hunter) in group.into_iter() {
            let mut position = if i == center_group_index {
                hunting_position_center(&hunter.position, &prey, &mut rng)
            } else {
                hunting_position_wing(&hunter.position, &prey, &mut rng)
            };
            limit_position(&mut position, lower_bound, upper_bound);
            let fitness = fitness_evaluator.calculate_fitness(&position);
            if fitness < hunter.fitness {
                prey = update_prey(&hunter.position, &prey, hunter.fitness - fitness, &mut rng);
                hunter.update_position(position, fitness);
            }
            new_hunters.push((j, hunter));
        }
    }
    new_hunters
}

fn calculate_tournament_size(lions: &[Lion]) -> usize {
    let kj: usize = lions.iter().map(|lion| lion.has_improved() as usize).sum();
    max(2, (kj as f64 / 2.0).ceil() as usize)
}

fn move_towards_safe_place(lion: &Lion, selected_lion: &Lion, mut rng: impl Rng) -> Vec<f64> {
    // let distance = euclidean_distance(&lion.position, &selected_lion.best_position);
    let r1 = lion.diff_position(&selected_lion.best_position);
    let r2 = perpendicular_position(&r1, &mut rng);
    (0..lion.position.len())
        .map(|i| {
            let u = rng.gen_range(-1.0, 1.0);
            let theta = rng.gen_range(-PI / 6.0, PI / 6.0);
            let r: f64 = rng.gen();
            let r1_i = r1[i];
            let r2_i = r2[i];
            let s_i = selected_lion.best_position[i] - lion.position[i];
            lion.position[i] + 2.0 * s_i * r * r1_i + u * theta.tan() * s_i * r2_i
        })
        .collect()
}

fn roam_pride(
    lion: &mut Lion,
    pride: &Vec<Lion>,
    roaming_percent: f64,
    lower_bound: f64,
    upper_bound: f64,
    fitness_evaluator: &FitnessEvaluator<f64>,
    mut rng: impl Rng,
) {
    let roaming_count = (roaming_percent * pride.len() as f64) as usize;
    let selected = seq::sample_iter(&mut rng, pride.iter(), roaming_count).unwrap();
    let roam_positions: Vec<_> = selected
        .into_iter()
        .map(|lion| &lion.best_position)
        .collect();
    for roam_position in roam_positions {
        let mut position: Vec<_> = lion.position
            .iter()
            .zip(roam_position)
            .map(|(p_i, r_i)| {
                let x = r_i - p_i;
                let theta = rng.gen_range(-PI / 6.0, PI / 6.0);
                p_i + x * theta.tan()
            })
            .collect();
        limit_position(&mut position, lower_bound, upper_bound);
        let fitness = fitness_evaluator.calculate_fitness(&position);
        lion.update_position(position, fitness);
    }
}

fn min_value(val1: f64, val2: f64) -> f64 {
    if val1 < val2 {
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

fn mutate_random(
    position: &Vec<f64>,
    mutation_probability: f64,
    lower_bound: f64,
    upper_bound: f64,
    mut rng: impl Rng,
) -> Vec<f64> {
    position
        .iter()
        .map(|pos_i| {
            let r: f64 = rng.gen();
            if r < mutation_probability {
                rng.gen_range(lower_bound, upper_bound)
            } else {
                *pos_i
            }
        })
        .collect()
}

fn mate(
    female: &Lion,
    males: &[Lion],
    config: &Config,
    fitness_evaluator: &FitnessEvaluator<f64>,
    mut rng: impl Rng,
) -> (Lion, Lion) {
    let normal = Normal::new(0.5, 0.1);
    let beta = normal.ind_sample(&mut rng);
    let males = vec![&rng.choose(males).unwrap().position];
    let (position1, position2) = uniform(&female.position, &males, beta);
    let position1 = mutate_random(
        &position1,
        config.mutation_probability,
        config.lower_bound,
        config.upper_bound,
        &mut rng,
    );
    let position2 = mutate_random(
        &position2,
        config.mutation_probability,
        config.lower_bound,
        config.upper_bound,
        &mut rng,
    );
    let fitness1 = fitness_evaluator.calculate_fitness(&position1);
    let fitness2 = fitness_evaluator.calculate_fitness(&position2);
    (
        Lion::new(position1, fitness1),
        Lion::new(position2, fitness2),
    )
}

// TODO: Replace with generic sort
fn sort_lions(population: &mut Vec<Lion>) {
    population
        .sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
}

fn defense_resident_male(
    old_males: Vec<Lion>,
    mut new_males: Vec<Lion>,
    mut males_in_pride: usize,
) -> (Vec<Lion>, Vec<Lion>) {
    let mut males = old_males;
    males.append(&mut new_males);
    sort_lions(&mut males);
    if males_in_pride >= males.len() {
        println!(
            "Limiting nomads. Wanted {}, but max was {}",
            males_in_pride,
            males.len()
        );
        males_in_pride = males.len();
    }
    let nomads = males.split_off(males_in_pride);
    (males, nomads)
}

fn defense_against_nomad_male<'a>(
    mut prides: Vec<Pride>,
    nomad: Nomad,
    mut rng: impl Rng,
) -> (Vec<Pride>, Nomad) {
    let nomad: Vec<Lion> = nomad
        .population
        .into_iter()
        .map(|nomad| {
            for pride in prides.iter_mut() {
                if rng.gen() {
                    continue;
                }
                let mut swapped: Option<Lion> = None;
                for lion in pride.population.iter() {
                    if lion.sex != Sex::Male {
                        continue;
                    }
                    if lion.fitness > nomad.fitness {
                        swapped = Some(lion.clone());
                        break;
                    }
                }
                if let Some(male) = swapped {
                    let index = pride
                        .population
                        .iter()
                        .position(|lion| lion == &male)
                        .unwrap();
                    pride.population.remove(index);
                    pride.population.push(nomad);
                    return male;
                }
            }
            nomad
        })
        .collect();
    (prides, Nomad { population: nomad })
}

fn migration(
    mut prides: Vec<Pride>,
    females_in_pride: usize,
    immigate_rate: f64,
    mut rng: impl Rng,
) -> (Vec<Pride>, Vec<Lion>) {
    let mut new_nomads: Vec<Lion> = Vec::new();
    for pride in prides.iter_mut() {
        let females: Vec<_> = pride
            .population
            .iter()
            .cloned()
            .filter(|lion| lion.sex == Sex::Female)
            .collect();
        println!("Females {} Max {}", females.len(), females_in_pride);
        let mut surplus: isize = females.len() as isize - females_in_pride as isize;
        let mut females_to_migrate = (immigate_rate * females_in_pride as f64) as isize;
        if surplus < 0 {
            println!("Surplas was negative {}", surplus);
            surplus = 0;
            females_to_migrate = 0;
        }
        let total_migrate = surplus + females_to_migrate;
        println!(
            "Total females migrating: {}. Surplus: {}. Immigrate: {}",
            total_migrate, surplus, females_to_migrate
        );
        let migrated_females = seq::sample_iter(&mut rng, females, total_migrate as usize).unwrap();
        for female in migrated_females.into_iter() {
            let index = pride
                .population
                .iter()
                .position(|lion| lion == &female)
                .unwrap();
            pride.population.remove(index);
            new_nomads.push(female);
        }
    }
    (prides, new_nomads)
}

fn assign_to_prides<'a>(
    prides: Vec<Pride>,
    mut nomad_females: Vec<Lion>,
    females_in_pride: usize,
    immigate_rate: f64,
    mut rng: impl Rng,
) -> Vec<Pride> {
    prides
        .into_iter()
        .map(|mut pride| {
            let missing_females = (immigate_rate * females_in_pride as f64) as usize;
            println!(
                "Max {} Missing {} nomad_left {}",
                females_in_pride,
                missing_females,
                nomad_females.len()
            );
            for _ in 0..missing_females {
                let i = rng.gen_range(0, nomad_females.len());
                let nomad_female = nomad_females.remove(i);
                pride.population.push(nomad_female);
            }
            pride
        })
        .collect::<Vec<Pride>>()
}

fn equilibrium(
    prides: Vec<Pride>,
    nomad: Nomad,
    config: &Config,
    mut rng: impl Rng,
) -> (Vec<Pride>, Nomad) {
    let mut sorted_nomads = nomad.population.into_iter().collect();
    sort_lions(&mut sorted_nomads);
    let (sorted_females, sorted_males) = sorted_nomads.into_iter().fold(
        (vec![], vec![]),
        |(mut females, mut males), lion| {
            if lion.sex == Sex::Female {
                females.push(lion);
            } else {
                males.push(lion);
            }
            (females, males)
        },
    );

    let mut females_for_prides = (config.population as f64 * (1.0 - config.nomad_percent)
        * config.sex_rate * config.immigate_rate) as usize;
    if females_for_prides >= sorted_females.len() {
        println!(
            "Too few female lions {}/{}",
            females_for_prides,
            sorted_females.len()
        );
        females_for_prides = sorted_females.len() - 1;
    }
    println!(
        "ffp {}, sorted_females {}",
        females_for_prides,
        sorted_females.len()
    );
    let prides = assign_to_prides(
        prides,
        sorted_females[..females_for_prides].to_vec(),
        config.females_in_pride(),
        config.immigate_rate,
        &mut rng,
    );

    let nomad_males_count =
        (config.population as f64 * config.nomad_percent * config.sex_rate) as usize;
    let nomad_females_count =
        (config.population as f64 * config.nomad_percent * (1.0 - config.sex_rate)) as usize;
    let mut population: Vec<Lion> = sorted_males[..nomad_males_count].to_vec();
    println!(
        "ffp {} nfc {} sf {}",
        females_for_prides,
        nomad_females_count,
        sorted_females.len()
    );
    population.extend(
        sorted_females[females_for_prides..females_for_prides + nomad_females_count].to_vec(),
    );
    (
        prides,
        Nomad {
            population: population.into_iter().collect(),
        },
    )
}

fn partition_on_sex(lions: Vec<Lion>) -> (Vec<Lion>, Vec<Lion>) {
    let mut males = vec![];
    let mut females = vec![];
    for lion in lions {
        match lion.sex {
            Sex::Male => males.push(lion),
            Sex::Female => females.push(lion),
            Sex::None => panic!("Tried to partiton lion with None as sex"),
        }
    }
    (males, females)
}

fn combine_population(prides: &Vec<Pride>, nomad: &Nomad) -> Vec<Lion> {
    let mut population = vec![];
    population.extend(nomad.population.iter().cloned());
    for pride in prides {
        population.extend(pride.population.iter().cloned());
    }
    population
}

fn print_info(prides: &Vec<Pride>, nomad: &Nomad) {
    println!("!!!!!!!! Population info !!!!!!!!");
    for (i, pride) in prides.iter().enumerate() {
        println!("Size before partition {}", pride.population.len());
        let (males, females) = partition_on_sex(pride.population.iter().cloned().collect());
        println!(
            "Pride [{}] Males {}, Females {}",
            i,
            males.len(),
            females.len()
        );
    }
    let (males, females) = partition_on_sex(nomad.population.iter().cloned().collect());
    println!("Nomad Males {} Females {}", males.len(), females.len());
    println!("!!!! END !!!!!");
}

fn run(config: Config, fitness_evaluator: &FitnessEvaluator<f64>) -> Vec<SolutionJSON> {
    let mut rng = thread_rng();
    let population = random_population(&config, &fitness_evaluator);
    let (mut nomad, mut prides) = partition_lions(&config, population);

    if config.verbose {
        println!("Males in pride {}", config.males_in_pride());
        println!("Females in pride {}", config.females_in_pride());
    }
    for i in 0..config.iterations {
        if config.verbose {
            println!(
                " ##### New iter [{}] Nomad {} ######",
                i,
                nomad.population.len(),
            );
            print_info(&prides, &nomad);
        }
        let hunters = find_hunters(&mut prides, &mut rng);
        let hunters = hunt(
            hunters,
            config.dimensions,
            config.lower_bound,
            config.upper_bound,
            &mut rng,
            &fitness_evaluator,
        );
        for (pride_index, hunter) in hunters {
            prides[pride_index].population.push(hunter);
        }
        prides = prides
            .into_iter()
            .map(|pride| {
                if config.verbose {
                    println!("---- New Pride! ----");
                }
                let lions: Vec<Lion> = pride.population.into_iter().collect();
                if config.verbose {
                    println!("Partion");
                }
                let (mut males, mut females) = partition_on_sex(lions.clone());
                if config.verbose {
                    println!("Females {} Males {}", females.len(), males.len());
                    println!("Roam");
                }
                for mut male in males.iter_mut() {
                    roam_pride(
                        &mut male,
                        &lions,
                        config.roaming_percent,
                        config.lower_bound,
                        config.upper_bound,
                        &fitness_evaluator,
                        &mut rng,
                    );
                }
                if config.verbose {
                    println!("Move safe");
                }
                females = females
                    .iter()
                    .cloned()
                    .map(|mut lion| {
                        let selected = {
                            let tournament_size = calculate_tournament_size(&females);
                            let (_, selected) =
                                tournament_selection(&females, tournament_size, &mut rng);
                            selected.clone()
                        };
                        let mut position = move_towards_safe_place(&lion, &selected, &mut rng);
                        limit_position(&mut position, config.lower_bound, config.upper_bound);
                        let fitness = fitness_evaluator.calculate_fitness(&position);
                        lion.update_position(position, fitness);
                        lion
                    })
                    .collect();
                if config.verbose {
                    println!("Mate");
                }
                let new_lions = females
                    .iter()
                    .flat_map(|female| {
                        let r: f64 = rng.gen();
                        if r < config.mating_probability {
                            let (lion1, lion2) =
                                mate(&female, &males, &config, &fitness_evaluator, &mut rng);
                            let mut lions = vec![lion1, lion2];
                            rng.shuffle(&mut lions);
                            lions[0].sex = Sex::Female;
                            lions[1].sex = Sex::Male;
                            lions
                        } else {
                            vec![]
                        }
                    })
                    .collect();
                let (new_males, mut new_females) = partition_on_sex(new_lions);
                let (males, new_nomad) =
                    defense_resident_male(males, new_males, config.males_in_pride());
                if config.verbose {
                    println!("Add");
                    println!("Males added: {}", males.len());
                }
                let mut population = males;
                if config.verbose {
                    println!(
                        "Old females added: {}, size: {}",
                        females.len(),
                        population.len()
                    );
                }
                population.append(&mut females);
                if config.verbose {
                    println!(
                        "New females added: {}, size: {}",
                        new_females.len(),
                        population.len()
                    );
                }
                population.append(&mut new_females);
                if config.verbose {
                    println!(
                        "New monads: {}, size: {}",
                        new_nomad.len(),
                        population.len()
                    );
                }
                nomad.population.extend(new_nomad);
                Pride {
                    population: population.into_iter().collect(),
                }
            })
            .collect();
        if config.verbose {
            println!("Prides completed");
            print_info(&prides, &nomad);
        }
        let best = nomad
            .population
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
            .clone();
        nomad = Nomad {
            population: nomad
                .population
                .into_iter()
                .map(|mut nomad_lion| {
                    roam_nomad(
                        &mut nomad_lion,
                        &best,
                        config.lower_bound,
                        config.upper_bound,
                        &fitness_evaluator,
                        &mut rng,
                    );
                    nomad_lion
                })
                .collect(),
        };
        let (mut males, mut females) = partition_on_sex(nomad.population.into_iter().collect());
        if config.verbose {
            println!(
                "Nomad males: {}. Nomad females: {}",
                males.len(),
                females.len()
            );
        }
        nomad.population = females
            .iter()
            .flat_map(|female| {
                let r: f64 = rng.gen();
                if r < config.mating_probability {
                    let (lion1, lion2) =
                        mate(&female, &males, &config, &fitness_evaluator, &mut rng);
                    let mut lions = vec![lion1, lion2];
                    rng.shuffle(&mut lions);
                    lions[0].sex = Sex::Female;
                    lions[1].sex = Sex::Male;
                    lions
                } else {
                    vec![]
                }
            })
            .collect();
        if config.verbose {
            println!("New nomad from mating: {}", nomad.population.len());
            println!(
                "Males added: {}, Females added: {}",
                males.len(),
                females.len()
            );
        }
        nomad.population.extend(males);
        nomad.population.extend(females);
        if config.verbose {
            println!("Pre defense {}", nomad.population.len());
        }
        let (new_prides, new_nomad) = defense_against_nomad_male(prides, nomad, &mut rng);
        if config.verbose {
            println!("Post defense {}", new_nomad.population.len());
        }
        prides = new_prides;
        nomad = new_nomad;
        if config.verbose {
            print_info(&prides, &nomad);
            println!("Migration");
        }

        let (new_prides, new_nomad) = migration(
            prides,
            config.females_in_pride(),
            config.immigate_rate,
            &mut rng,
        );
        if config.verbose {
            println!("New nomad after migration {}", new_nomad.len());
        }
        nomad.population.extend(new_nomad);
        prides = new_prides;
        if config.verbose {
            print_info(&prides, &nomad);
            println!("Equilibrium");
        }
        let (new_prides, new_nomad) = equilibrium(prides, nomad, &config, &mut rng);
        prides = new_prides;
        nomad = new_nomad;
        if config.verbose {
            print_info(&prides, &nomad);
        }
        fitness_evaluator
            .sampler
            .population_sample_single(i, &combine_population(&prides, &nomad));
    }
    fitness_evaluator
        .sampler
        .population_sample_single(config.iterations, &combine_population(&prides, &nomad));
    vec![]
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
            dimensions: 2,
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
    fn min_value_returns_min() {
        let val = min_value(0.2, 0.3);

        assert_eq!(val, 0.2);
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
        let population = create_population();

        let (nomad, prides) = partition_lions(&config, population);

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
    fn finds_female_in_pride() {
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0], 1.0, Sex::Male),
            create_lion_with_sex(vec![2.0, 2.0], 2.0, Sex::Female),
            create_lion_with_sex(vec![3.0, 3.0], 3.0, Sex::Male),
        ];
        let pride = Pride {
            population: population.iter().cloned().collect(),
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
            population: population.into_iter().collect(),
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
        let mut prides = vec![
            Pride {
                population: population[..2].iter().cloned().collect(),
            },
            Pride {
                population: population[2..4].iter().cloned().collect(),
            },
            Pride {
                population: population[4..6].iter().cloned().collect(),
            },
        ];
        let rng = create_rng();

        let hunters = find_hunters(&mut prides, rng);

        assert_eq!(
            hunters,
            vec![
                (0, population[1].clone()),
                (1, population[3].clone()),
                (2, population[5].clone()),
            ]
        );
    }

    #[test]
    fn calculates_prey_position_correctly() {
        let population = vec![
            create_lion_with_sex(vec![2.0, 3.0], 3.0, Sex::Female),
            create_lion_with_sex(vec![7.0, 2.0], 3.0, Sex::Female),
        ];
        let hunters = population.iter().collect();

        let prey_position = calculate_prey_position(&hunters, 2);

        assert_eq!(prey_position, vec![9.0 / 2.0, 5.0 / 2.0]);
    }

    #[test]
    fn partitions_hunters_randomly() {
        let population = vec![
            create_lion_with_sex(vec![2.0, 3.0], 3.0, Sex::Female),
            create_lion_with_sex(vec![7.0, 1.0], 2.0, Sex::Female),
            create_lion_with_sex(vec![3.0, 6.0], 7.0, Sex::Female),
            create_lion_with_sex(vec![2.0, 3.0], 4.0, Sex::Female),
        ];
        let hunters: Vec<(usize, Lion)> = population.into_iter().map(|l| (0, l)).collect();
        let rng = create_rng();

        let groups = group_hunters(hunters, rng);

        let lions_in_groups: Vec<_> = groups.iter().flat_map(|g| g).collect();
        assert_eq!(groups.len(), 3);
        assert_eq!(lions_in_groups.len(), 4);
    }

    #[test]
    fn selects_group_with_highest_fitness() {
        let groups = vec![
            vec![(0, create_lion_with_sex(vec![3.0, 6.0], 6.0, Sex::Female))],
            vec![
                (1, create_lion_with_sex(vec![2.0, 3.0], 3.0, Sex::Female)),
                (1, create_lion_with_sex(vec![2.0, 3.0], 5.0, Sex::Female)),
            ],
            vec![(2, create_lion_with_sex(vec![7.0, 1.0], 2.0, Sex::Female))],
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
        let rng = create_rng();

        let position = move_towards_safe_place(&mut lion, &selected_lion, rng);

        // More or less hoping that this is the correct result
        assert_eq!(
            position,
            vec![2.161088604263452, 3.5674186416279423, 4.1529584756243425]
        );
    }

    #[test]
    fn counts_improved_lions() {
        let mut lion1 = Lion::new(vec![0.1, 0.2, 0.31], 0.31);
        let lion2 = Lion::new(vec![0.1, 0.2, 0.32], 0.32);
        let mut lion3 = Lion::new(vec![0.1, 0.2, 0.33], 0.33);
        lion1.update_position(vec![0.1, 0.2, 0.21], 0.21);
        lion3.update_position(vec![0.1, 0.2, 0.23], 0.23);

        let tournament_size = calculate_tournament_size(&vec![lion1, lion2, lion3]);

        assert_eq!(tournament_size, 2);
    }

    #[test]
    fn moves_lion_with_roam_nomad() {
        let mut lion = create_lion_with_sex(vec![0.2, 0.1, 0.3], 0.3, Sex::Male);
        let best = create_lion_with_sex(vec![0.2, 0.1, 0.0], 0.1, Sex::Male);
        let mut rng = create_rng();
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);

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
        let mut rng = create_rng();
        let sampler = create_sampler();
        let fitness_evaluator = create_evaluator(&sampler);

        let original_position = lion.position.clone();
        roam_pride(
            &mut lion,
            &population,
            0.4,
            -10.0,
            10.0,
            &fitness_evaluator,
            &mut rng,
        );

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
        let old_males = vec![
            create_lion_with_sex(vec![0.5, 0.5], 0.5, Sex::Male),
            create_lion_with_sex(vec![0.3, 0.3], 0.3, Sex::Male),
            create_lion_with_sex(vec![0.2, 0.2], 0.2, Sex::Male),
        ];
        let new_males = vec![
            create_lion_with_sex(vec![0.4, 0.4], 0.4, Sex::Male),
            create_lion_with_sex(vec![0.6, 0.6], 0.6, Sex::Male),
            create_lion_with_sex(vec![0.1, 0.1], 0.1, Sex::Male),
        ];

        let (pride_males, nomads) = defense_resident_male(old_males, new_males, 3);

        // Checking fitness only as several mutable referens does not work good in Rust
        let pride_males_fitness: Vec<_> = pride_males.iter().map(|l| l.fitness).collect();
        assert_eq!(pride_males_fitness, vec![0.1, 0.2, 0.3]);
        let nomads_fitness: Vec<_> = nomads.iter().map(|l| l.fitness).collect();
        assert_eq!(nomads_fitness, vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn defends_against_nomad_male() {
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
                population: population[..2].iter().cloned().collect(),
            },
            Pride {
                population: population[2..4].iter().cloned().collect(),
            },
            Pride {
                population: population[4..6].iter().cloned().collect(),
            },
        ];
        let nomad_males = vec![
            create_lion_with_sex(vec![0.5, 0.3], 0.8, Sex::Male),
            create_lion_with_sex(vec![2.5, 1.3], 1.4, Sex::Male),
            create_lion_with_sex(vec![10.0, 10.0], 10.0, Sex::Male),
        ];
        let worst_nomad = nomad_males[2].clone();
        let nomad = Nomad {
            population: nomad_males.into_iter().collect(),
        };
        let rng = create_rng();

        let (prides, nomad) = defense_against_nomad_male(prides, nomad, rng);

        assert_eq!(nomad.population.len(), 3);
        let pride_population: Vec<_> = prides.iter().flat_map(|pride| &pride.population).collect();
        assert_eq!(pride_population.len(), 6);
        let females: Vec<_> = pride_population
            .iter()
            .filter(|lion| lion.sex == Sex::Female)
            .collect();
        assert_eq!(females.len(), 3);
        assert!(nomad.population.contains(&worst_nomad));
    }

    #[test]
    fn migrates_surplus() {
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
                population: population[..3].iter().cloned().collect(),
            },
            Pride {
                population: population[3..6].iter().cloned().collect(),
            },
        ];
        let rng = create_rng();

        let (prides, new_nomads) = migration(prides, 1, 0.0, rng);

        assert_eq!(prides[0].population.len(), 3);
        assert_eq!(prides[1].population.len(), 2);
        assert_eq!(new_nomads.len(), 1);
    }

    #[test]
    fn migrates_surplus_and_random() {
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0], 1.0, Sex::Male),
            create_lion_with_sex(vec![2.0, 2.0], 2.0, Sex::Female),
            create_lion_with_sex(vec![3.0, 3.0], 3.0, Sex::Male),
            create_lion_with_sex(vec![4.0, 4.0], 4.0, Sex::Female),
            create_lion_with_sex(vec![5.0, 5.0], 5.0, Sex::Male),
            create_lion_with_sex(vec![6.0, 6.0], 6.0, Sex::Female),
        ];
        let prides = vec![Pride {
            population: population.into_iter().collect(),
        }];
        let rng = create_rng();

        let (prides, new_nomads) = migration(prides, 1, 0.5, rng);

        assert_eq!(prides[0].population.len(), 4);
        assert_eq!(new_nomads.len(), 2);
    }

    #[test]
    fn test_assigns_nomads_to_prides() {
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0], 1.0, Sex::Male),
            create_lion_with_sex(vec![2.0, 2.0], 2.0, Sex::Male),
            create_lion_with_sex(vec![3.0, 3.0], 3.0, Sex::Male),
            create_lion_with_sex(vec![4.0, 4.0], 4.0, Sex::Female),
            create_lion_with_sex(vec![5.0, 5.0], 5.0, Sex::Male),
            create_lion_with_sex(vec![6.0, 6.0], 6.0, Sex::Male),
            create_lion_with_sex(vec![7.0, 7.0], 7.0, Sex::Male),
            create_lion_with_sex(vec![8.0, 8.0], 8.0, Sex::Female),
            create_lion_with_sex(vec![9.0, 9.0], 9.0, Sex::Male),
            create_lion_with_sex(vec![10.0, 10.0], 10.0, Sex::Female),
        ];
        let prides = vec![
            Pride {
                population: population[..5].iter().cloned().collect(),
            },
            Pride {
                population: population[5..].iter().cloned().collect(),
            },
        ];
        let nomads = vec![
            create_lion_with_sex(vec![0.1, 0.3], 0.1, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.2, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.3, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.4, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.5, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.6, Sex::Female),
        ];
        let rng = create_rng();

        let prides = assign_to_prides(prides, nomads.iter().cloned().collect(), 5, 0.5, rng);

        assert_eq!(prides.len(), 2);
        assert_eq!(prides[0].population.len(), 7);
        assert_eq!(prides[1].population.len(), 7);
    }

    #[test]
    fn test_equilibrium() {
        let population = vec![
            create_lion_with_sex(vec![1.0, 1.0], 1.0, Sex::Male),
            create_lion_with_sex(vec![2.0, 2.0], 2.0, Sex::Male),
            create_lion_with_sex(vec![3.0, 3.0], 3.0, Sex::Male),
            create_lion_with_sex(vec![4.0, 4.0], 4.0, Sex::Female),
            create_lion_with_sex(vec![5.0, 5.0], 5.0, Sex::Male),
            create_lion_with_sex(vec![6.0, 6.0], 6.0, Sex::Male),
            create_lion_with_sex(vec![7.0, 7.0], 7.0, Sex::Male),
            create_lion_with_sex(vec![8.0, 8.0], 8.0, Sex::Female),
            create_lion_with_sex(vec![9.0, 9.0], 9.0, Sex::Male),
            create_lion_with_sex(vec![10.0, 10.0], 10.0, Sex::Male),
        ];
        let prides = vec![
            Pride {
                population: population[..5].iter().cloned().collect(),
            },
            Pride {
                population: population[5..].iter().cloned().collect(),
            },
        ];
        let nomads = [
            create_lion_with_sex(vec![0.1, 0.3], 0.1, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.2, Sex::Male),
            create_lion_with_sex(vec![0.2, 0.3], 0.3, Sex::Male),
            create_lion_with_sex(vec![0.2, 0.3], 0.4, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.5, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.6, Sex::Female),
            create_lion_with_sex(vec![0.2, 0.3], 0.7, Sex::Male),
            create_lion_with_sex(vec![0.2, 0.3], 0.8, Sex::Female),
        ];
        let nomad = Nomad {
            population: nomads.iter().cloned().collect(),
        };

        let config = Config {
            iterations: 100,
            population: 15,
            upper_bound: 1.0,
            lower_bound: -1.0,
            dimensions: 2,
            prides: 2,
            nomad_percent: 0.2,
            roaming_percent: 0.2,
            mutation_probability: 0.2,
            sex_rate: 0.8,
            mating_probability: 0.3,
            immigate_rate: 0.3,
        };
        let rng = create_rng();

        let (prides, nomad) = equilibrium(prides, nomad, &config, rng);

        assert_eq!(prides[0].population.len(), 6);
        assert_eq!(prides[1].population.len(), 6);
        assert_eq!(nomad.population.len(), 2);
    }

    #[test]
    fn test_mate() {
        let female = create_lion_with_sex(vec![0.5, 0.8], 1.0, Sex::Female);
        let males = vec![
            create_lion_with_sex(vec![1.5, 1.8], 2.0, Sex::Male),
            // create_lion_with_sex(vec![2.5, 2.8], 3.0),
        ];
        let config = create_config();
        let sampler = create_sampler();
        let evaluator = create_evaluator(&sampler);
        let rng = create_rng();

        let (offspring1, offspring2) = mate(&female, &males, &config, &evaluator, rng);

        assert_eq!(
            offspring1.position,
            vec![0.8038002274057862, 0.09380622409375228]
        );
        assert_eq!(
            offspring2.position,
            vec![1.1961997725942137, 1.4961997725942138]
        );
    }
}
