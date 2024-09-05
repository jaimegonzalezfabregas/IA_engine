use std::sync::mpsc::Sender;

use ia_engine::{
    dual::Dual,
    trainer::{DataPoint, Trainer},
};
use image::{imageops::FilterType::Triangle, DynamicImage, GenericImageView, ImageReader};


use crate::{tiler, TILE_COUNT};

pub fn train_thread(tx: Sender<([f32; TILE_COUNT * 5], Option<f32>)>) {
    let mut trainer = Trainer::new(tiler, |e| {
        e.map(|p| Dual::new_full(p.get_real().min(1.).max(0.), &p.get_gradient()))
    }, ());
    tx.send((trainer.get_model_params(), None)).unwrap();

       let img = ImageReader::open("assets/rust.png")
        .unwrap()
        .decode()
        .unwrap();

    let mut dataset_complexity = 1;

    loop {
        let not_reached_local_min = trainer.train_step(&dataset_provider(&img, dataset_complexity));

        if !not_reached_local_min {
            println!("complexify");
            dataset_complexity += 1;
        }

        tx.send((trainer.get_model_params(), trainer.get_last_cost()))
            .unwrap();
        // for semi_pixels in pixels.array_chunks::<90>() {
        //     println!("{:?}",semi_pixels);
        //     trainer.train_step(&Vec::from(semi_pixels));
        // }
    }
}

fn dataset_provider(og_img: &DynamicImage, dataset_complexity: usize)->Vec<DataPoint<{TILE_COUNT*5},2,3>> {
 

    let img = og_img
        .resize_exact(100, 100, Triangle)
        .blur(1. / dataset_complexity as f32);

    img
        .pixels()
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
