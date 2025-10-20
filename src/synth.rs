#[allow(dead_code)]
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use serde::{Serialize, Serializer};
use serde_json::Value;
use std::collections::HashSet;

const EPS: f32 = 1e-12;
const DIRECTIONS: [(i32, i32); 8] = [
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
    (1, -1),
    (-1, -1),
    (1, 1),
    (-1, 1),
];

const ALPHANUM_ARR: [u8; 62] = *b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const SYMBOL_MAX_LEN: usize = 3;

fn serialize_symbol<S>(symbol: &[u8; 3], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // Convert bytes to string (ignore zeros)
    let s = symbol
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as char)
        .collect::<String>();
    serializer.serialize_str(&s)
}

#[derive(Serialize, Clone)]
struct Vertex {
    #[serde(serialize_with = "serialize_symbol")]
    symbol: [u8; SYMBOL_MAX_LEN],
    id: u16,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    neighbours: Vec<u16>,
}

#[derive(Serialize)]
struct Edge<'a> {
    id: usize,
    relationship: [u16; 2],
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    logic_label: &'a str,
}

pub(crate) struct Plain {
    regions: Vec<Vec<Option<u16>>>,
    vertices: Vec<Vertex>,
    rng: ThreadRng,
    region_size: u16,
    vertex_radius: u16,
    max_vertex_count: u16,
    max_neighbours_count: u16,
    weighted_index: WeightedIndex<f32>,
    used_symbol: HashSet<[u8; 3]>,
}

impl Vertex {
    fn new(id: u16, symbol: [u8; 3], x: f32, y: f32, radius: f32) -> Self {
        Self {
            id,
            symbol,
            x,
            y,
            width: radius * 2.0,
            height: radius * 2.0,
            neighbours: Vec::new(),
        }
    }
}

static mut PAIN_NUM: usize = 0;

impl Plain {
    pub fn new(
        regions_cols: usize,
        regions_rows: usize,
        region_size: u16,
        vertex_radius: u16,
        max_vertex_count: u16,
        max_neighbours_count: u16,
        weights: &[f32],
    ) -> Self {
        let dist = WeightedIndex::new(weights).unwrap();
        let mut plain = Self {
            regions: vec![vec![None; regions_cols]; regions_rows],
            vertices: Vec::new(),
            rng: rand::rng(),
            max_vertex_count,
            max_neighbours_count,
            region_size,
            vertex_radius,
            weighted_index: dist,
            used_symbol: HashSet::new(),
        };

        let pos = plain.rand_pos();
        let symbol_len = plain.rng.random_range(1..=SYMBOL_MAX_LEN);
        let start = Vertex::new(
            0,
            plain.rand_symbol(symbol_len as u8),
            (plain.region_size * pos.0 as u16) as f32,
            (plain.region_size * pos.1 as u16) as f32,
            plain.vertex_radius as f32,
        );
        plain.regions[pos.1][pos.0] = Some(0);
        plain.vertices.push(start);
        plain
    }

    fn rand_symbol(&mut self, len: u8) -> [u8; 3] {
        let mut symbol = [0u8; 3];
        loop {
            for i in 0..len as usize {
                symbol[i] = ALPHANUM_ARR[self.rng.random_range(0..ALPHANUM_ARR.len())];
            }
            if !self.used_symbol.contains(&symbol) {
                self.used_symbol.insert(symbol);
                break;
            }
        }
        symbol
    }

    fn rand_pos(&mut self) -> (usize, usize) {
        (
            self.rng.random_range(0..self.regions[0].len()),
            self.rng.random_range(0..self.regions.len()),
        )
    }

    fn put_next_vertex(&mut self, prev_id: usize) -> Option<u16> {
        let x_step = self.weighted_index.sample(&mut self.rng) as i32;
        let y_step = self.weighted_index.sample(&mut self.rng) as i32;
        if x_step == 0 && y_step == 0 {
            return None;
        }

        let dir = self.rng.random_range(0..=7);
        let (dx, dy) = DIRECTIONS[dir];

        let new_x = (self.vertices[prev_id].x / self.region_size as f32) as i32 + (dx * x_step);
        let new_y = (self.vertices[prev_id].y / self.region_size as f32) as i32 + (dy * y_step);

        if new_x < 0
            || new_y < 0
            || new_x >= self.regions[0].len() as i32
            || new_y >= self.regions.len() as i32
        {
            return None;
        }

        let grid_x = new_x as usize;
        let grid_y = new_y as usize;

        if let Some(existing_id) = self.regions[grid_y][grid_x] {
            if !self.vertices[prev_id].neighbours.contains(&existing_id) {
                self.vertices[prev_id].neighbours.push(existing_id);
            }
            if !self.vertices[existing_id as usize]
                .neighbours
                .contains(&(prev_id as u16))
            {
                self.vertices[existing_id as usize]
                    .neighbours
                    .push(prev_id as u16);
            }
            return None;
        }

        let x_offset = self
            .rng
            .random_range(0..self.region_size - self.vertex_radius * 2)
            / 4;
        let y_offset = self
            .rng
            .random_range(0..self.region_size - self.vertex_radius * 2)
            / 4;

        let id = self.vertices.len() as u16;
        let symbol_len = self.rng.random_range(1..=3);

        let new_vertex = Vertex::new(
            id,
            self.rand_symbol(symbol_len),
            grid_x as f32 * self.region_size as f32 + x_offset as f32,
            grid_y as f32 * self.region_size as f32 + y_offset as f32,
            self.vertex_radius as f32,
        );

        self.regions[grid_y][grid_x] = Some(id);
        self.vertices.push(new_vertex);

        self.vertices[prev_id].neighbours.push(id);
        self.vertices[id as usize].neighbours.push(prev_id as u16);

        Some(id)
    }

    fn _run_sim(&mut self, current: usize) {
        for _ in 0..self.max_neighbours_count {
            if self.vertices.len() < self.max_vertex_count as usize {
                if let Some(next) = self.put_next_vertex(current) {
                    self._run_sim(next as usize);
                }
            }
        }
    }

    pub fn run_sim(&mut self) {
        self._run_sim(0);
    }

    pub(crate) fn dump(&self, file_path: &str) {
        let json = self.create_json();
        std::fs::write(file_path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    pub(crate) fn dump_many(plains: Vec<Self>, file_path: &str) {
        let mut results: Vec<Value> = Vec::with_capacity(plains.len());
        for plain in plains {
            results.push(plain.create_json())
        }
        std::fs::write(file_path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }

    fn create_json(&self) -> Value {
        let img_width = self.regions[0].len() as f32 * self.region_size as f32;
        let img_height = self.regions.len() as f32 * self.region_size as f32;

        let mut edges = Vec::new();

        for v in self.vertices.iter() {
            for &n in &v.neighbours {
                if v.id < n {
                    let neighbour = &self.vertices[n as usize];

                    let dx = neighbour.x - v.x;
                    let dy = neighbour.y - v.y;
                    let dist = (dx * dx + dy * dy).sqrt();

                    if dist < EPS {
                        continue;
                    }

                    let ux = dx / dist;
                    let uy = dy / dist;

                    let start_x = v.x + ux * v.width / 2.0;
                    let start_y = v.y + uy * v.height / 2.0;
                    let end_x = neighbour.x - ux * neighbour.width / 2.0;
                    let end_y = neighbour.y - uy * neighbour.height / 2.0;

                    let min_x = start_x.min(end_x);
                    let min_y = start_y.min(end_y);
                    let max_x = start_x.max(end_x);
                    let max_y = start_y.max(end_y);

                    let edge = Edge {
                        id: edges.len(),
                        relationship: [v.id, n],
                        x: min_x,
                        y: min_y,
                        width: max_x - min_x,
                        height: max_y - min_y,
                        logic_label: "edge",
                    };
                    edges.push(edge);
                }
            }
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;

        for v in &self.vertices {
            min_x = min_x.min(v.x - v.width / 2.0);
            min_y = min_y.min(v.y - v.height / 2.0);
        }

        for e in &edges {
            min_x = min_x.min(e.x);
            min_y = min_y.min(e.y);
        }

        let vertices_norm: Vec<_> = self
            .vertices
            .iter()
            .map(|v| {
                let x_norm = (v.x - v.width / 2.0 - min_x) / img_width;
                let y_norm = (v.y - v.height / 2.0 - min_y) / img_height;
                Vertex {
                    symbol: v.symbol,
                    id: v.id,
                    x: x_norm,
                    y: y_norm,
                    width: v.width / img_width,
                    height: v.height / img_height,
                    neighbours: v.neighbours.clone(),
                }
            })
            .collect();

        let edges_norm: Vec<_> = edges
            .into_iter()
            .map(|mut e| {
                e.x = (e.x - min_x) / img_width;
                e.y = (e.y - min_y) / img_height;
                e.width /= img_width;
                e.height /= img_height;
                e
            })
            .collect();

        let json = serde_json::json!({
            "filename": format!("Synth_Graph_{}.png", unsafe { PAIN_NUM + 1 }),
            "img_width": img_width as u32,
            "img_height": img_height as u32,
            "vertices": vertices_norm,
            "connections": edges_norm,
            "graph_type": "Undirected-graph"
        });

        unsafe {
            PAIN_NUM += 1;
        }

        json
    }
}
