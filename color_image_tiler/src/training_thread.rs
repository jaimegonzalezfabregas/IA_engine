use std::sync::mpsc::Sender;

use ia_engine::{
    dual::Dual,
    trainer::{DataPoint, Trainer},
};
use image::{DynamicImage, GenericImageView, ImageReader};
use rand::seq::SliceRandom;

use crate::{tiler, TILE_COUNT};

pub fn train_thread(tx: Sender<([f32; TILE_COUNT * 5], (Option<f32>, usize))>) {
    let mut rng = rand::thread_rng();
    let mut trainer = Trainer::new(
        tiler,
        |e| e.map(|p| Dual::new_full(p.get_real().min(1.).max(0.), &p.get_gradient())),
        |params| {
        //    ( Dual::from(1.) / params
        //         .array_chunks::<5>()
        //         .flat_map(|[x1, y1, _, _, _]| {
        //             params
        //                 .array_chunks::<5>()
        //                 .map(|[x, y, _, _, _]| (x1, y1, x, y))
        //                 .collect::<Vec<_>>()
        //         })
        //         .fold(Dual::cero(), |acc, (x1, y1, x2, y2)| {
        //             let dx = *x1 - *x2;
        //             let dy = *y1 - *y2;
        //             acc + (dx * dx) + (dy * dy)
        //         }))*300.
            Dual::cero()
                
        },
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
