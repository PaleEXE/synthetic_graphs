use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use serde::Serialize;

#[derive(Serialize, Clone)]
struct Vertex {
    symbol: char,
    id: u16,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    neighbours: Vec<u16>,
}

#[derive(Serialize)]
struct Edge {
    id: usize,
    relationship: [u16; 2],
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

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

struct Plain {
    regions: Vec<Vec<Option<u16>>>,
    vertices: Vec<Vertex>,
    rng: ThreadRng,
    region_size: u16,
    vertex_radius: u16,
    max_vertex_count: u16,
    max_neighbours_count: u16,
    weighted_index: WeightedIndex<f32>,
}

impl Vertex {
    fn new(id: u16, symbol: char, x: f32, y: f32, radius: f32) -> Self {
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

impl Plain {
    fn new(
        regions_cols: usize,
        regions_rows: usize,
        region_size: u16,
        vertex_radius: u16,
        max_vertex_count: u16,
        max_neighbours_count: u16,
        weights: &[f32],
    ) -> Self {
        let dist = WeightedIndex::new(weights).unwrap();
        let mut region = Self {
            regions: vec![vec![None; regions_cols]; regions_rows],
            vertices: Vec::new(),
            rng: rand::rng(),
            max_vertex_count,
            max_neighbours_count,
            region_size,
            vertex_radius,
            weighted_index: dist,
        };

        let pos = region.rand_pos();
        let start = Vertex::new(
            0,
            'A',
            (region.region_size * pos.0 as u16) as f32,
            (region.region_size * pos.1 as u16) as f32,
            region.vertex_radius as f32,
        );
        region.regions[pos.1][pos.0] = Some(0);
        region.vertices.push(start);
        region
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
            / 2;
        let y_offset = self
            .rng
            .random_range(0..self.region_size - self.vertex_radius * 2)
            / 2;

        let id = self.vertices.len() as u16;
        let new_vertex = Vertex::new(
            id,
            (b'A' + (id % 26) as u8) as char,
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

    fn run_sim(&mut self, current: usize) {
        for _ in 0..self.max_neighbours_count {
            if self.vertices.len() < self.max_vertex_count as usize {
                if let Some(next) = self.put_next_vertex(current) {
                    self.run_sim(next as usize);
                }
            }
        }
    }

    // In your dump function - FIXED for object detection:
    fn dump(&self, file_path: &str) {
        let img_width = self.regions[0].len() as f32 * self.region_size as f32;
        let img_height = self.regions.len() as f32 * self.region_size as f32;

        let mut edges = Vec::new();

        // Create edges with proper bounding boxes
        for v in self.vertices.iter() {
            for &n in &v.neighbours {
                if v.id < n {
                    let right = &self.vertices[n as usize];

                    // Edge bounding box that contains the entire line
                    let min_x = v.x.min(right.x);
                    let min_y = v.y.min(right.y);
                    let max_x = v.x.max(right.x);
                    let max_y = v.y.max(right.y);

                    let edge = Edge {
                        id: edges.len(),
                        relationship: [v.id, n],
                        x: min_x / img_width,  // normalized top-left
                        y: min_y / img_height, // normalized top-left
                        width: (max_x - min_x) / img_width,
                        height: (max_y - min_y) / img_height,
                    };
                    edges.push(edge);
                }
            }
        }

        // Normalize vertices (already using center coordinates)
        let vertices_norm: Vec<_> = self
            .vertices
            .iter()
            .map(|v| Vertex {
                symbol: v.symbol,
                id: v.id,
                x: (v.x - v.width / 2.0) / img_width, // convert center to top-left for bbox
                y: (v.y - v.height / 2.0) / img_height, // convert center to top-left for bbox
                width: v.width / img_width,
                height: v.height / img_height,
                neighbours: v.neighbours.clone(),
            })
            .collect();

        let json = serde_json::json!({
            "img_width": img_width as u32,
            "img_height": img_height as u32,
            "vertices": vertices_norm,
            "edges": edges
        });

        std::fs::write(file_path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }
}

fn main() {
    let mut plain = Plain::new(10, 10, 40, 12, 12, 8, &[0.0, 0.5, 0.8, 0.2, 0.0, 0.0]);
    plain.run_sim(0);
    plain.dump("graph.json");
}
