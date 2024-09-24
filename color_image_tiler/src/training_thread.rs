use std::{array, sync::mpsc::Sender};

use ia_engine::{
    simd_arr::{hybrid_simd::HybridSimd, SimdArr},
    trainer::{DataPoint, Trainer},
};
use image::{imageops::sample_bilinear, DynamicImage, GenericImageView, ImageReader};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{tiler, TrainerComunicationCodes, TILE_COUNT};

fn max_speed_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    let ret = array::from_fn(|i| match i % 5 {
        0 | 1 => (params[i] + vector[i]).min(1.).max(0.),
        _ => params[i] + vector[i],
    });

    ret
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
        let pixels = dataset_provider(&img, 100);

        // println!("{:?}", pixels);

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

    println!("{:?}", trainer.get_model_params());
}

fn dataset_provider(img: &DynamicImage, count: usize) -> Vec<DataPoint<{ TILE_COUNT * 5 }, 2, 3>> {
    let mut rng = rand::thread_rng();
    (0..count)
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
        .collect()
}
