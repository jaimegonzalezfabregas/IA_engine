use ia_engine::trainer::DataPoint;
use image::{imageops::sample_bilinear, DynamicImage, ImageReader};
use rand::Rng;

use crate::{seed::Seed, TILE_COUNT, TILE_COUNT_SQRT};

pub struct DatasetSampleService {
    img: DynamicImage,
    scale: f32,
}

impl DatasetSampleService {
    pub fn new(path: &str, scale: f32) -> Self {
        let img = ImageReader::open(path).unwrap().decode().unwrap();
        Self { img, scale }
    }

    fn sample(&self, raw_x: f32, raw_y: f32) -> Option<[f32; 3]> {
        let x = (raw_x - 0.5) * self.scale + 0.5;
        let y = (raw_y - 0.5) * self.scale + 0.5;

        // println!("{raw_x} {raw_y} {x} {y}");

        sample_bilinear(&self.img, x, y).map(|c| {
            [
                c.0[0] as f32 / u8::MAX as f32,
                c.0[1] as f32 / u8::MAX as f32,
                c.0[2] as f32 / u8::MAX as f32,
            ]
        })
    }

    pub fn get(
        &self,
        count: usize,
        params: &[f32; TILE_COUNT * 5],
    ) -> Vec<DataPoint<{ TILE_COUNT * 5 }, 2, 3>> {
        let mut rng = rand::thread_rng();
        let random_samples: Vec<_> = (0..count)
            .map(|_| (rng.gen(), rng.gen()))
            .map(|(x, y)| (x, y, self.sample(x, y)))
            .filter(|(_, _, x)| x.is_some())
            .map(|(x, y, c)| DataPoint {
                input: [x, y],
                output: c.unwrap(),
            })
            .collect();

        let mut deliberate_samples: Vec<DataPoint<{ TILE_COUNT * 5 }, 2, 3>> = vec![];

        let seeds = params.array_chunks::<5>().collect::<Vec<_>>();

        for (i, seed) in seeds.iter().enumerate() {
            let [relative_x, relative_y, _, _, _] = seed;
            let og = Seed::new(*relative_x, *relative_y, i);

            for (dx, dy) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                // (1, 1),
                // (-1, -1),
                // (-1, 1),
                // (1, -1),
            ] {
                let neigh_cell_x = og.cell_x as isize + dx;
                let neigh_cell_y = og.cell_y as isize + dy;

                if neigh_cell_x < 0 || neigh_cell_x >= TILE_COUNT_SQRT as isize {
                    continue;
                }
                if neigh_cell_y < 0 || neigh_cell_y >= TILE_COUNT_SQRT as isize {
                    continue;
                }

                let j = neigh_cell_x as usize * TILE_COUNT_SQRT + neigh_cell_y as usize;

                let [other_relative_x, other_relative_y, _, _, _] = seeds[j];

                let other = Seed::new(*other_relative_x, *other_relative_y, j);

                let mid_x = (other.x + og.x) / 2.;
                let mid_y = (other.y + og.y) / 2.;

                let c = self.sample(mid_x, mid_y);

                if let Some(c) = c {
                    deliberate_samples.push(DataPoint {
                        input: [mid_x, mid_y],
                        output: c,
                    });
                }
            }
        }

        random_samples
            .into_iter()
            .chain(deliberate_samples)
            .collect()
    }
}
