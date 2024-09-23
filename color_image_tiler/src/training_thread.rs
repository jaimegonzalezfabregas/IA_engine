use std::{array, sync::mpsc::Sender};

use ia_engine::{
    dual::Dual,
    simd_arr::{hybrid_simd::HybridSimd, SimdArr},
    trainer::{DataPoint, Trainer},
};
use image::{DynamicImage, GenericImageView, ImageReader};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{tiler, TrainerComunicationCodes, TILE_COUNT};

/*
fn enforce_separation(
    p1: (f32, f32),
    p1_old: (f32, f32),
    p2: (f32, f32),
    p2_old: (f32, f32),
) -> (f32, f32, f32, f32, bool) {
    // Define the minimum separation distance
    const MIN_DISTANCE: f32 = 0.006; // Adjust as needed

    // Calculate the distance between p1 and p2
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    let dist = (dx * dx + dy * dy).sqrt();

    // If the distance is less than the minimum distance, we need to adjust
    let ret = if dist < MIN_DISTANCE {
        // Calculate the amount of adjustment needed
        let adjustment = MIN_DISTANCE * 1.05 - dist;

        let (unit_dx, unit_dy) = if dist == 0. {
            let dx_old = p1_old.0 - p2_old.0;
            let dy_old = p1_old.1 - p2_old.1;
            let dist_old = (dy_old * dy_old + dy_old * dy_old).sqrt();

            if dist_old == 0. {
                (0., 1.)
            } else {
                (dx_old / dist_old, dy_old / dist_old)
            }
        } else {
            (dx / dist, dy / dist)
        };

        // Compute the unit vector in the direction from p2 to p1

        // Move p1 and p2 apart by half the adjustment distance in opposite directions
        let p1_new = (
            p1.0 + unit_dx * adjustment / 2.,
            p1.1 + unit_dy * adjustment / 2.,
        );

        let p2_new = (
            p2.0 - unit_dx * adjustment / 2.,
            p2.1 - unit_dy * adjustment / 2.,
        );

        (p1_new.0, p1_new.1, p2_new.0, p2_new.1, true)
    } else {
        // If the distance is already sufficient, return the original positions
        (p1.0, p1.1, p2.0, p2.1, false)
    };

    ret
}


fn colision_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    let mut new_params = max_speed_param_translator(params, vector);

    let mut rep = true;

    while rep {
        rep = false;

        new_params.map(|p| p.min(1.).max(0.));

        let mut cases = vec![];

        for i in 0..TILE_COUNT {
            let pos_1 = (new_params[i * 5], new_params[i * 5 + 1]);
            let pos_1_old = (params[i * 5], params[i * 5 + 1]);

            for j in (i + 1)..TILE_COUNT {
                let pos_2 = (new_params[j * 5], new_params[j * 5 + 1]);
                let pos_2_old = (params[j * 5], params[j * 5 + 1]);

                cases.push((i, j, pos_1, pos_1_old, pos_2, pos_2_old));
            }
        }

        cases.shuffle(&mut rng);

        for (i, j, pos_1, pos_1_old, pos_2, pos_2_old) in cases {
            let (a, b, c, d, any_colision) = enforce_separation(pos_1, pos_1_old, pos_2, pos_2_old);
            rep = rep || any_colision;
            new_params[i * 5] = a.min(1.).max(0.);
            new_params[i * 5 + 1] = b.min(1.).max(0.);
            new_params[j * 5] = c.min(1.).max(0.);
            new_params[j * 5 + 1] = d.min(1.).max(0.);
        }
    }

    new_params
}
*/

fn max_speed_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    let ret = array::from_fn(|i| match i % 5 {
        0 | 1 => (params[i] + vector[i]).max(0.).min(1.),
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

fn train_work<S: SimdArr<{ TILE_COUNT * 5 }>>(
    tx: Option<Sender<TrainerComunicationCodes<([f32; TILE_COUNT * 5], (Option<f32>, usize))>>>,
    max_iterations: Option<usize>,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut trainer: Trainer<_, _, _, _, S, _, _, _> =
        Trainer::new(tiler, tiler, max_speed_param_translator, ());
    if let Some(ref tx) = tx {
        tx.send(TrainerComunicationCodes::Msg((
            trainer.get_model_params(),
            (None, 0),
        )))
        .unwrap();
    }

    let img = ImageReader::open(
        "/home/jaime/Desktop/projects/IA_engine/color_image_tiler/assets/rust.png",
    )
    .unwrap()
    .decode()
    .unwrap();
    println!("get dataset");

    let mut pixels = dataset_provider(&img);

    let mut local_minimum_count = 0;
    let mut iterations = 0;
    while local_minimum_count < 100 {
        pixels.shuffle(&mut rng);

        for semi_pixels in pixels.array_chunks::<100>() {
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

            if !trainer.train_step(&Vec::from(semi_pixels)) {
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
    }

    println!("training done");

    println!("{:?}", trainer.get_model_params());
}

fn dataset_provider(og_img: &DynamicImage) -> Vec<DataPoint<{ TILE_COUNT * 5 }, 2, 3>> {
    let img = og_img; //.resize_exact(100, 100, Triangle);
                      // .blur(1. / dataset_complexity as f32);

    img.pixels()
        .map(|(x, y, c)| DataPoint {
            input: [
                x as f32 / img.dimensions().0 as f32,
                1. - y as f32 / img.dimensions().1 as f32,
            ],
            output: [
                c.0[0] as f32 / u8::MAX as f32,
                c.0[1] as f32 / u8::MAX as f32,
                c.0[2] as f32 / u8::MAX as f32,
            ],
        })
        .collect()
}

extern crate test;

#[cfg(test)]
mod tests {
    use std::thread;

    use crate::TILE_COUNT;

    use super::test::Bencher;

    use ia_engine::simd_arr::{dense_simd::DenseSimd, hybrid_simd::HybridSimd, SimdArr};

    use super::train_work;

    fn bench<S: SimdArr<{ TILE_COUNT * 5 }>>() {
        let train_builder = thread::Builder::new()
            .name("train_thread".into())
            .stack_size(2 * 1024 * 1024 * 1024);

        let _ = train_builder
            .spawn(|| train_work::<S>(None, Some(100)))
            .unwrap()
            .join();
    }

    #[bench]
    fn bench_dense(b: &mut Bencher) {
        b.iter(|| bench::<DenseSimd<_>>());
    }
    #[bench]
    fn bench_hybrid_1(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 1>>());
    }
    #[bench]
    fn bench_hybrid_2(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 2>>());
    }
    #[bench]
    fn bench_hybrid_3(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 3>>());
    }
    #[bench]
    fn bench_hybrid_4(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 4>>());
    }
    #[bench]
    fn bench_hybrid_10(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 10>>());
    }
    #[bench]
    fn bench_hybrid_100(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 100>>());
    }
}
