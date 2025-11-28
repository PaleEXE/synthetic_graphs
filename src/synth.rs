#![allow(dead_code)]
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

const NUMS: &[u8] = b"0123456789";
const UPPER_ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const LOWER_ALPHA: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const ALPHANUM: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const SYMBOL_MAX_LEN: usize = 1;

fn serialize_symbol<S>(symbol: &[u8; SYMBOL_MAX_LEN], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let s = symbol
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as char)
        .collect::<String>();
    serializer.serialize_str(&s)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SymbolPool {
    Num,
    Alpha,
    UpperAlpha,
    LowerAlpha,
    AlphaNum,
}

struct VectorDelta {
    length: f32,
    dx: f32,
    dy: f32,
}

#[derive(Serialize)]
struct Rectangle {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
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
    #[serde(flatten)]
    bound: Rectangle,
    cost: Option<u16>,
    logic_label: &'a str,
}

pub(crate) struct Plain<'a> {
    regions: Vec<Vec<Option<u16>>>,
    vertices: Vec<Vertex>,
    rng: ThreadRng,
    region_size: u16,
    vertex_radius: u16,
    max_vertex_count: u16,
    max_neighbours_count: u16,
    weighted_index: WeightedIndex<f32>,
    used_symbol: HashSet<[u8; SYMBOL_MAX_LEN]>,
    image_name: &'a str,
    with_cost: bool,
    symbol_pool: SymbolPool,
}

impl SymbolPool {
    fn chars_pool(self) -> &'static [u8] {
        match self {
            SymbolPool::Num => NUMS,
            SymbolPool::Alpha => ALPHA,
            SymbolPool::UpperAlpha => UPPER_ALPHA,
            SymbolPool::LowerAlpha => LOWER_ALPHA,
            SymbolPool::AlphaNum => ALPHANUM,
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "num" => SymbolPool::Num,
            "alpha" => SymbolPool::Alpha,
            "upperalpha" => SymbolPool::UpperAlpha,
            "loweralpha" => SymbolPool::LowerAlpha,
            "alphanum" => SymbolPool::AlphaNum,
            _ => panic!("Invalid symbol pool: {}", s),
        }
    }
}

impl Vertex {
    fn new(id: u16, symbol: [u8; SYMBOL_MAX_LEN], x: f32, y: f32, radius: f32) -> Self {
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

    fn get_distance(&self, other: &Self) -> VectorDelta {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dist = (dx * dx + dy * dy).sqrt();
        VectorDelta {
            length: dist,
            dx,
            dy,
        }
    }

    fn get_edge_rect(&self, other: &Self, vd: &VectorDelta) -> Rectangle {
        let ux = vd.dx / vd.length;
        let uy = vd.dy / vd.length;
        let start_x = self.x + ux * self.width / 2.0;
        let start_y = self.y + uy * self.height / 2.0;
        let end_x = other.x - ux * other.width / 2.0;
        let end_y = other.y - uy * other.height / 2.0;
        let min_x = start_x.min(end_x);
        let min_y = start_y.min(end_y);
        let max_x = start_x.max(end_x);
        let max_y = start_y.max(end_y);
        Rectangle {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
}

static mut PAIN_NUM: usize = 0;
pub static mut SIM_NUM: usize = 0;
impl<'a> Plain<'a> {
    pub fn new(
        regions_cols: usize,
        regions_rows: usize,
        region_size: u16,
        vertex_radius: u16,
        max_vertex_count: u16,
        max_neighbours_count: u16,
        weights: &[f32],
        image_name: &'a str,
        with_cost: bool,
        symbol_pool: SymbolPool,
    ) -> Self {
        let dist = WeightedIndex::new(weights.to_vec()).unwrap();
        let rng = rand::rng();
        let max_vertex_count = if symbol_pool == SymbolPool::Num {
            max_vertex_count.min(9)
        } else {
            max_vertex_count
        };
        let mut plain = Self {
            regions: vec![vec![None; regions_cols]; regions_rows],
            vertices: Vec::new(),
            rng,
            max_vertex_count,
            max_neighbours_count,
            region_size,
            vertex_radius,
            weighted_index: dist,
            used_symbol: HashSet::new(),
            image_name,
            with_cost,
            symbol_pool,
        };
        let pos = plain.rand_pos();
        let start = Vertex::new(
            0,
            plain.rand_symbol(),
            (plain.region_size * pos.0 as u16) as f32,
            (plain.region_size * pos.1 as u16) as f32,
            plain.vertex_radius as f32,
        );
        plain.regions[pos.1][pos.0] = Some(0);
        plain.vertices.push(start);
        plain
    }

    fn rand_symbol(&mut self) -> [u8; SYMBOL_MAX_LEN] {
        let mut symbol = [0u8; SYMBOL_MAX_LEN];
        let pool_chars = self.symbol_pool.chars_pool();
        let pool_len = pool_chars.len();
        for _ in 0..(pool_len.max(1)) {
            let idx = self.rng.random_range(0..pool_len);
            symbol[0] = pool_chars[idx];
            if !self.used_symbol.contains(&symbol) {
                self.used_symbol.insert(symbol);
                return symbol;
            }
        }
        let idx = self.rng.random_range(0..pool_len);
        symbol[0] = pool_chars[idx];
        symbol
    }

    fn rand_pos(&mut self) -> (usize, usize) {
        (
            self.rng.random_range(0..self.regions[0].len()),
            self.rng.random_range(0..self.regions.len()),
        )
    }

    fn put_next_vertex(&mut self, prev_id: usize) -> Option<u16> {
        const ATTEMPTS: usize = 24;
        let prev = &self.vertices[prev_id];
        let prev_grid_x = (prev.x / self.region_size as f32) as i32;
        let prev_grid_y = (prev.y / self.region_size as f32) as i32;
        let cols = self.regions[0].len() as i32;
        let rows = self.regions.len() as i32;
        for _ in 0..ATTEMPTS {
            let x_step = self.weighted_index.sample(&mut self.rng) as i32;
            let y_step = self.weighted_index.sample(&mut self.rng) as i32;
            if x_step == 0 && y_step == 0 {
                continue;
            }
            let dir = self.rng.random_range(0..DIRECTIONS.len());
            let (dx, dy) = DIRECTIONS[dir];
            let new_x = prev_grid_x + dx * x_step;
            let new_y = prev_grid_y + dy * y_step;
            if self.inside_plain(new_x, new_y) {
                continue;
            }
            let grid_x = new_x as usize;
            let grid_y = new_y as usize;
            if let Some(existing_id) = self.regions[grid_y][grid_x] {
                if existing_id != prev_id as u16
                    && !self.is_path_blocked(
                        &self.vertices[prev_id],
                        &self.vertices[existing_id as usize],
                    )
                {
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
                    return Some(existing_id);
                }
                continue;
            }
            let max_offset_base = if self.region_size > self.vertex_radius * 2 {
                (self.region_size - self.vertex_radius * 2) as usize
            } else {
                1usize
            };
            let x_offset = (self.rng.random_range(0..max_offset_base) / 3) as f32;
            let y_offset = (self.rng.random_range(0..max_offset_base) / 3) as f32;
            let id = self.vertices.len() as u16;
            let candidate = Vertex::new(
                id,
                self.rand_symbol(),
                grid_x as f32 * self.region_size as f32 + x_offset,
                grid_y as f32 * self.region_size as f32 + y_offset,
                self.vertex_radius as f32,
            );
            if self.is_path_blocked(&self.vertices[prev_id], &candidate) {
                continue;
            }
            self.regions[grid_y][grid_x] = Some(id);
            self.vertices.push(candidate);
            self.vertices[prev_id].neighbours.push(id);
            self.vertices[id as usize].neighbours.push(prev_id as u16);
            return Some(id);
        }
        let max_search = (cols.max(rows)) as i32;
        for r in 1..=max_search {
            for gx in (prev_grid_x - r)..=(prev_grid_x + r) {
                for gy in (prev_grid_y - r)..=(prev_grid_y + r) {
                    if gx < 0 || gy < 0 || gx >= cols || gy >= rows {
                        continue;
                    }
                    let ux = gx as usize;
                    let uy = gy as usize;
                    if self.regions[uy][ux].is_some() {
                        continue;
                    }
                    let id = self.vertices.len() as u16;
                    let max_offset_base = if self.region_size > self.vertex_radius * 2 {
                        (self.region_size - self.vertex_radius * 2) as usize
                    } else {
                        1usize
                    };
                    let x_offset = (self.rng.random_range(0..max_offset_base) / 3) as f32;
                    let y_offset = (self.rng.random_range(0..max_offset_base) / 3) as f32;
                    let candidate = Vertex::new(
                        id,
                        self.rand_symbol(),
                        ux as f32 * self.region_size as f32 + x_offset,
                        uy as f32 * self.region_size as f32 + y_offset,
                        self.vertex_radius as f32,
                    );
                    if self.is_path_blocked(&self.vertices[prev_id], &candidate) {
                        continue;
                    }
                    self.regions[uy][ux] = Some(id);
                    self.vertices.push(candidate);
                    self.vertices[prev_id].neighbours.push(id);
                    self.vertices[id as usize].neighbours.push(prev_id as u16);
                    return Some(id);
                }
            }
        }
        for (idx, v) in self.vertices.iter().enumerate() {
            if v.id == prev_id as u16 {
                continue;
            }
            if !self.is_path_blocked(&self.vertices[prev_id], v) {
                let eid = v.id;
                if !self.vertices[prev_id].neighbours.contains(&eid) {
                    self.vertices[prev_id].neighbours.push(eid);
                }
                if !self.vertices[idx].neighbours.contains(&(prev_id as u16)) {
                    self.vertices[idx].neighbours.push(prev_id as u16);
                }
                return Some(eid);
            }
        }
        None
    }

    // FIX: Updated to check for obstructions along the edge-to-edge segment
    // between the start and end vertices, ensuring no path passes through the
    // physical space occupied by other nodes.
    fn is_path_blocked(&self, start: &Vertex, end: &Vertex) -> bool {
        // 1. Calculate center-to-center vector (SE) and its length
        let sx = start.x;
        let sy = start.y;
        let ex = end.x;
        let ey = end.y;
        let dx = ex - sx;
        let dy = ey - sy;
        let len_sq = dx * dx + dy * dy;

        if len_sq <= EPS {
            return true; // Vertices are identical or too close
        }

        let len = len_sq.sqrt();

        // 2. Calculate edge-to-edge segment (S'E')
        let rs = start.width * 0.5;
        let re = end.width * 0.5;

        // Unit vector u along SE
        let ux = dx / len;
        let uy = dy / len;

        // Start point on the edge (S')
        let s_prime_x = sx + ux * rs;
        let s_prime_y = sy + uy * rs;

        // End point on the edge (E')
        let e_prime_x = ex - ux * re;
        let e_prime_y = ey - uy * re;

        // Edge-to-edge segment vector d'
        let d_prime_x = e_prime_x - s_prime_x;
        let d_prime_y = e_prime_y - s_prime_y;
        let len_prime_sq = d_prime_x * d_prime_x + d_prime_y * d_prime_y;

        if len_prime_sq <= EPS {
            // The edge-to-edge segment is negligible or the vertices are already touching/overlapping.
            return false;
        }

        // 3. Iterate through all existing vertices to check for blocking
        for v in &self.vertices {
            // Skip the start and end vertices of the current path check
            if v.id == start.id || v.id == end.id {
                continue;
            }

            // Vector S'V
            let s_prime_v_x = v.x - s_prime_x;
            let s_prime_v_y = v.y - s_prime_y;

            // Projection parameter t' on S'E'
            // t' = (S'V . S'E') / |S'E'|^2
            let t_prime = (s_prime_v_x * d_prime_x + s_prime_v_y * d_prime_y) / len_prime_sq;

            // Clamping t' to the segment [0, 1]
            let t_clamped = if t_prime < 0.0 {
                0.0
            } else if t_prime > 1.0 {
                1.0
            } else {
                t_prime
            };

            // Closest point P on S'E' segment
            let closest_x = s_prime_x + t_clamped * d_prime_x;
            let closest_y = s_prime_y + t_clamped * d_prime_y;

            // Vector PV (from closest point to V's center)
            let vx = v.x - closest_x;
            let vy = v.y - closest_y;
            let dist_sq = vx * vx + vy * vy;

            // Blocking vertex radius R_v
            let radius = (v.width.max(v.height)) * 0.5;

            // Collision check: is distance <= R_v ?
            if dist_sq <= (radius + EPS) * (radius + EPS) {
                return true;
            }
        }
        false
    }

    fn inside_plain(&self, new_x: i32, new_y: i32) -> bool {
        new_x < 0
            || new_y < 0
            || new_x >= self.regions[0].len() as i32
            || new_y >= self.regions.len() as i32
    }

    fn get_offset(&mut self) -> u16 {
        let base = if self.region_size > self.vertex_radius * 2 {
            (self.region_size - self.vertex_radius * 2) as usize
        } else {
            1usize
        };
        (self.rng.random_range(0..base) / 3) as u16
    }

    pub fn run_sim(&mut self) {
        const MAX_SIM_ITER: usize = 100_000;
        let mut stack: Vec<usize> = vec![0usize];
        let mut iter_count: usize = 0;
        while let Some(current) = stack.pop() {
            iter_count += 1;
            if iter_count > MAX_SIM_ITER {
                break;
            }
            if self.vertices.len() >= self.max_vertex_count as usize {
                break;
            }
            let mut created_any = false;
            for _ in 0..self.max_neighbours_count {
                if self.vertices.len() >= self.max_vertex_count as usize {
                    break;
                }
                if let Some(next) = self.put_next_vertex(current) {
                    created_any = true;
                    let newly_created_index = self.vertices.len().saturating_sub(1);
                    if next as usize == newly_created_index {
                        stack.push(next as usize);
                    }
                }
            }
            if stack.is_empty() {
                break;
            }
            if !created_any && stack.is_empty() {
                break;
            }
        }
    }

    pub(crate) fn dump(&mut self, file_path: &str) {
        let json = self.create_json();
        std::fs::write(file_path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    pub(crate) fn dump_many(plains: Vec<Self>, file_path: &str) {
        let mut results: Vec<Value> = Vec::with_capacity(plains.len());
        for mut plain in plains {
            results.push(plain.create_json())
        }
        std::fs::write(file_path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }

    fn create_json(&mut self) -> Value {
        let img_width = self.regions[0].len() as f32 * self.region_size as f32;
        let img_height = self.regions.len() as f32 * self.region_size as f32;
        let mut edges = Vec::new();
        for v in self.vertices.iter() {
            for &n in &v.neighbours {
                if v.id < n {
                    let neighbour = &self.vertices[n as usize];
                    let vd = v.get_distance(neighbour);
                    if vd.length < EPS {
                        continue;
                    }
                    let bound = v.get_edge_rect(neighbour, &vd);
                    let cost = if self.with_cost {
                        Some(vd.length as u16 / 50 + self.rng.random_range(1..=3) as u16)
                    } else {
                        None
                    };
                    let edge = Edge {
                        id: edges.len(),
                        relationship: [v.id, n],
                        bound,
                        cost,
                        logic_label: "edge",
                        // ...
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
            min_x = min_x.min(e.bound.x);
            min_y = min_y.min(e.bound.y);
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
                e.bound.x = (e.bound.x - min_x) / img_width;
                e.bound.y = (e.bound.y - min_y) / img_height;
                e.bound.width /= img_width;
                e.bound.height /= img_height;
                e
            })
            .collect();
        let sim_num: usize = unsafe { SIM_NUM };
        let pain_num: usize = unsafe { PAIN_NUM };
        let image_num = format!("{:0width$}", pain_num, width = sim_num.to_string().len());
        let file_name = format!("{}_{}.png", self.image_name, image_num);
        let json = serde_json::json!({
            "filename": file_name,
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
