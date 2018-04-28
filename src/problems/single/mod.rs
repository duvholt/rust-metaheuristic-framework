pub mod cec2014;
pub mod misc;
use std::collections::HashMap;

pub fn add_test_suite(test_suites: &mut HashMap<&'static str, Vec<String>>) {
    test_suites.insert(
        "single",
        vec![
            "high-elliptic".to_string(),
            "bent-cigar".to_string(),
            "discus".to_string(),
            "rosenbrock".to_string(),
            "ackley".to_string(),
            "weierstrass".to_string(),
            "griewank".to_string(),
            "rastrigin".to_string(),
            "schwefel".to_string(),
            "katsuura".to_string(),
            "happycat".to_string(),
            "hgbat".to_string(),
            "griewank-rosenbrock".to_string(),
            "expanded-schaffer6".to_string(),
            "zakharov".to_string(),
            "hyper-ellipsoid".to_string(),
            "moved-hyper-ellipsoid".to_string(),
            "easom".to_string(),
            "sphere".to_string(),
        ],
    );
}
