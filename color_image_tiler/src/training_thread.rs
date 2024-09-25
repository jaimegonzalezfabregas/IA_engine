use std::{array, sync::mpsc::Sender};

use ia_engine::{
    simd_arr::{hybrid_simd::HybridSimd, SimdArr},
    trainer::{DataPoint, Trainer},
};
use image::{imageops::sample_bilinear, DynamicImage, ImageReader};
use rand::Rng;

use crate::{seed::Seed, tiler, TrainerComunicationCodes, TILE_COUNT, TILE_COUNT_SQRT};

fn max_speed_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    array::from_fn(|i| (params[i] + vector[i]).max(0.).min(1.))
}

pub fn train_thread(
    tx: Sender<TrainerComunicationCodes<([f32; TILE_COUNT * 5], (Option<f32>, usize))>>,
    max_iterations: Option<usize>,
) {
    train_work::<HybridSimd<_, { TILE_COUNT * 2 }>>(Some(tx), max_iterations);
    // train_work::<DenseSimd<_>>(Some(tx), max_iterations);
}

pub(crate) fn train_work<S: SimdArr<{ TILE_COUNT * 5 }>>(
    tx: Option<Sender<TrainerComunicationCodes<([f32; TILE_COUNT * 5], (Option<f32>, usize))>>>,
    max_iterations: Option<usize>,
) {
    let mut trainer: Trainer<_, _, _, _, S, _, _, _> =
        Trainer::new(tiler, tiler, max_speed_param_translator, ());
    if let Some(ref tx) = tx {
        tx.send(TrainerComunicationCodes::Msg((
            trainer.get_model_params(),
            (None, 0),
        )))
        .unwrap();
    }

    let img = ImageReader::open("assets/circle.png")
        .unwrap()
        .decode()
        .unwrap();
    println!("get dataset");

    let mut local_minimum_count = 0;
    let mut iterations = 0;
    while local_minimum_count < 100 {
        let pixels = dataset_provider(&img, 100, &trainer.get_model_params());

        iterations += 1;

        if max_iterations
            .map(|max_it| iterations >= max_it)
            .unwrap_or(false)
        {
            if let Some(ref tx) = tx {
                tx.send(TrainerComunicationCodes::Die).unwrap();
            }
            return;
        }

        if !trainer.train_step(&pixels) {
            local_minimum_count += 1;


            trainer

        } else {
            local_minimum_count = 0;
        }

        if let Some(ref tx) = tx {
            tx.send(TrainerComunicationCodes::Msg((
                trainer.get_model_params(),
                (trainer.get_last_cost(), local_minimum_count),
            )))
            .unwrap();
        }
    }

    println!("training done");

    // println!("{:?}", trainer.get_model_params());
}

fn dataset_provider(
    img: &DynamicImage,
    count: usize,
    params: &[f32; TILE_COUNT * 5],
) -> Vec<DataPoint<{ TILE_COUNT * 5 }, 2, 3>> {
    let mut rng = rand::thread_rng();
    let random_samples: Vec<_> = (0..count)
        .map(|_| (rng.gen(), rng.gen()))
        .map(|(x, y)| (x, y, sample_bilinear(img, x, y).unwrap()))
        .map(|(x, y, c)| DataPoint {
            input: [x, y],
            output: [
                c.0[0] as f32 / u8::MAX as f32,
                c.0[1] as f32 / u8::MAX as f32,
                c.0[2] as f32 / u8::MAX as f32,
            ],
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
            (1, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
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

            let c = sample_bilinear(img, mid_x, mid_y);

            if let Some(c) = c {
                deliberate_samples.push(DataPoint {
                    input: [mid_x, mid_y],
                    output: [
                        c.0[0] as f32 / u8::MAX as f32,
                        c.0[1] as f32 / u8::MAX as f32,
                        c.0[2] as f32 / u8::MAX as f32,
                    ],
                });
            }
        }
    }

    random_samples
        .into_iter()
        .chain(deliberate_samples)
        .collect()
}
