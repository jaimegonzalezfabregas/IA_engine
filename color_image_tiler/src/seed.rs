use crate::{PARTICLE_FREEDOM, TILE_COUNT_SQRT};

pub struct Seed {
    pub x: f32,
    pub y: f32,
    pub cell_x: usize,
    pub cell_y: usize,
}

impl Seed {
    pub fn new(relative_x: f32, relative_y: f32, i: usize) -> Self {
        let cell_x = i / TILE_COUNT_SQRT;
        let cell_y = i % TILE_COUNT_SQRT;

        let x = (cell_x as f32 + relative_x * PARTICLE_FREEDOM as f32) / TILE_COUNT_SQRT as f32;
        let y = (cell_y as f32 + relative_y * PARTICLE_FREEDOM as f32) / TILE_COUNT_SQRT as f32;

        Seed { cell_x, cell_y, x, y }
    }
}
