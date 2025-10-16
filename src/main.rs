use crate::synth::Plain;
use rand::Rng;
use std::time::Instant;
use dotenvy::dotenv;
use std::env;

mod synth;

fn main() {
    dotenv().ok(); // load .env file automatically

    let num_of_sim: usize = env::var("NUM_OF_SIM").unwrap().parse().unwrap();
    let min_regions: usize = env::var("MIN_REGIONS_RANGE").unwrap().parse().unwrap();
    let max_regions: usize = env::var("MAX_REGIONS_RANGE").unwrap().parse().unwrap();
    let min_region_size: u16 = env::var("MIN_REGION_SIZE").unwrap().parse().unwrap();
    let max_region_size: u16 = env::var("MAX_REGION_SIZE").unwrap().parse().unwrap();
    let min_vertex_count: u16 = env::var("MIN_VERTEX_COUNT").unwrap().parse().unwrap();
    let max_vertex_count: u16 = env::var("MAX_VERTEX_COUNT").unwrap().parse().unwrap();
    let min_neighbour_count: u16 = env::var("MIN_NEIGHBOUR_COUNT").unwrap().parse().unwrap();
    let max_neighbour_count: u16 = env::var("MAX_NEIGHBOUR_COUNT").unwrap().parse().unwrap();
    let output_json = env::var("OUTPUT_JSON").unwrap_or("synth_graphs.json".into());

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

        let mut plain = Plain::new(
            cols,
            rows,
            size,
            v_size,
            v_num,
            neighbours,
            &[0.8, 0.8, 0.4, 0.2, 0.0, 0.0],
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

    println!("\nPress Enter to exit...");
    let mut buffer = String::new();
    let _ = std::io::stdin().read_line(&mut buffer);
}
