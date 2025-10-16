use crate::synth::Plain;
use rand::Rng;
use std::time::Instant;

mod synth;

const NUM_OF_SIM: usize = 5000;
const MIN_REGIONS_RANGE: usize = 6;
const MAX_REGIONS_RANGE: usize = 10;
const MIN_REGION_SIZE: u16 = 32;
const MAX_REGION_SIZE: u16 = 64;
const MIN_VERTEX_COUNT: u16 = 4;
const MAX_VERTEX_COUNT: u16 = 16;
const MIN_NEIGHBOUR_COUNT: u16 = 2;
const MAX_NEIGHBOUR_COUNT: u16 = 12;

fn main() {
    let mut results: Vec<Plain> = Vec::with_capacity(NUM_OF_SIM);
    let mut rng = rand::rng();

    let start = Instant::now();

    for i in 1..=NUM_OF_SIM {
        let cols = rng.random_range(MIN_REGIONS_RANGE..MAX_REGIONS_RANGE);
        let rows = rng.random_range(MIN_REGIONS_RANGE..MAX_REGIONS_RANGE);
        let size = rng.random_range(MIN_REGION_SIZE..MAX_REGION_SIZE);
        let v_size = rng.random_range(MIN_REGION_SIZE-1..size) / 2 - 5;
        let v_num = rng.random_range(MIN_VERTEX_COUNT..MAX_VERTEX_COUNT);
        let neighbours = rng.random_range(MIN_NEIGHBOUR_COUNT..MAX_NEIGHBOUR_COUNT);

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
        println!("[SIM] is done: {i}")
    }

    Plain::dump_many(results, "synth_graphs.json");
    let duration = start.elapsed();

    println!(
        "Simulations completed: {}\nTotal time: {:.2?}\nAverage time per simulation: {:.4?}",
        NUM_OF_SIM,
        duration,
        duration / NUM_OF_SIM as u32
    );
}
