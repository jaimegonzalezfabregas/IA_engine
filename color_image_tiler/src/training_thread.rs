use std::{array, sync::mpsc::Sender};

use ia_engine::trainer::{default_extra_cost, DataPoint, Trainer};
use image::{DynamicImage, GenericImageView, ImageReader};
use rand::seq::SliceRandom;

use crate::{tiler, TILE_COUNT};

pub fn train_thread(tx: Sender<([f32; TILE_COUNT * 5], (Option<f32>, usize))>) {
    let mut rng = rand::thread_rng();
    let mut trainer = Trainer::new(
        tiler,
        |params: &[f32; TILE_COUNT * 5], gradient: &[f32; TILE_COUNT * 5]| {
            let new_params = array::from_fn(|i| params[i] + gradient[i]);

            new_params.map(|p| p.min(1.).max(0.))
        },
        default_extra_cost,
        (),
    );
    tx.send((trainer.get_model_params(), (None, 0))).unwrap();
    println!("read_image");

    let img = ImageReader::open("assets/rust.png")
        .unwrap()
        .decode()
        .unwrap();
    println!("get dataset");

    let mut pixels = dataset_provider(&img);

    let mut local_minimum_count = 0;
    while local_minimum_count < 100 {
        println!("shuffling");

        pixels.shuffle(&mut rng);
        println!("shuffled");

        for semi_pixels in pixels.array_chunks::<10>() {
            if !trainer.train_step(&Vec::from(semi_pixels)) {
                local_minimum_count += 1;
            } else {
                local_minimum_count = 0;
            }
            tx.send((
                trainer.get_model_params(),
                (trainer.get_last_cost(), local_minimum_count),
            ))
            .unwrap();
        }
    }

    println!("training done");
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
