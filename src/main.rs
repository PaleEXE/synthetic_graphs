use crate::synth::{Plain, SIM_NUM, SymbolPool};
use dotenvy::dotenv;
use rand::Rng;
use std::env;
use std::time::Instant;

mod synth;

fn parse_env<T: std::str::FromStr>(key: &str) -> T {
    env::var(key)
        .unwrap_or_else(|_| panic!("Environment variable {key} not set"))
        .parse()
        .unwrap_or_else(|_| panic!("Failed to parse {key}"))
}

fn parse_env_vec<T: std::str::FromStr>(key: &str) -> Vec<T> {
    env::var(key)
        .unwrap_or_else(|_| panic!("Environment variable {key} not set"))
        .split(',')
        .map(|x| {
            x.trim()
                .parse::<T>()
                .unwrap_or_else(|_| panic!("Failed to parse float"))
        })
        .collect()
}

fn main() {
    dotenv().ok();

    let num_of_sim: usize = parse_env("NUM_OF_SIM");
    unsafe {
        SIM_NUM = num_of_sim;
    }

    let min_regions: usize = parse_env("MIN_REGIONS_RANGE");
    let max_regions: usize = parse_env("MAX_REGIONS_RANGE");
    let min_region_size: u16 = parse_env("MIN_REGION_SIZE");
    let max_region_size: u16 = parse_env("MAX_REGION_SIZE");
    let min_vertex_count: u16 = parse_env("MIN_VERTEX_COUNT");
    let max_vertex_count: u16 = parse_env("MAX_VERTEX_COUNT");
    let min_neighbour_count: u16 = parse_env("MIN_NEIGHBOUR_COUNT");
    let max_neighbour_count: u16 = parse_env("MAX_NEIGHBOUR_COUNT");

    let steps_weight: Vec<f32> = parse_env_vec("STEPS_WEIGHT");
    let symbols_pools: Vec<String> = parse_env_vec("SYMBOLS_POOLS");

    let symbols_pools = symbols_pools
        .iter()
        .map(|s| SymbolPool::from_str(&s.to_lowercase()))
        .collect::<Vec<SymbolPool>>();

    let with_cost = env::var("WITH_COST").unwrap_or_default().to_uppercase() == "TRUE";
    let output_json = env::var("OUTPUT_JSON").unwrap_or_else(|_| "synth_graphs.json".into());
    let image_name = env::var("IMAGE_NAME").unwrap_or_else(|_| "Synth_Graphs".into());

    let mut results: Vec<Plain> = Vec::with_capacity(num_of_sim);
    let mut rng = rand::rng();
    let start = Instant::now();

    for i in 1..=num_of_sim {
        let cols = rng.random_range(min_regions..max_regions);
        let rows = rng.random_range(min_regions..max_regions);
        let size = rng.random_range(min_region_size..max_region_size);
        let v_size = rng.random_range(min_region_size - 1..size) / 2 - 5;
        let v_num = rng.random_range(min_vertex_count..max_vertex_count);
        let neighbours = rng.random_range(min_neighbour_count..max_neighbour_count);

        let rand_pool = rng.random_range(0..symbols_pools.len());
        let symbols_pool = symbols_pools[rand_pool];

        let mut plain = Plain::new(
            cols,
            rows,
            size,
            v_size,
            v_num,
            neighbours,
            &steps_weight,
            &image_name,
            with_cost,
            symbols_pool,
        );

        plain.run_sim();
        results.push(plain);
        println!("[SIM] done: {i}");
    }

    Plain::dump_many(results, &output_json);

    let duration = start.elapsed();
    println!(
        "Simulations completed: {}\nTotal time: {:.2?}\nAverage time per simulation: {:.4?}",
        num_of_sim,
        duration,
        duration / num_of_sim as u32
    );
}
