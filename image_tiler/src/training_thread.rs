use std::sync::mpsc::Sender;

use ia_engine::{
    dual::Dual,
    trainer::{DataPoint, Trainer},
};
use image::{imageops::FilterType::Triangle, GenericImageView, ImageReader};

use crate::{tiler, TILE_COUNT};

pub fn train_thread(tx: Sender<[f32; { TILE_COUNT * 5 }]>) {
    let mut trainer = Trainer::new(tiler, |e| {
        e.map(|p| Dual::new_full(p.get_real().max(0.).min(1.), p.get_gradient()))
    });

    let img = ImageReader::open("assets/voronoi.png")
        .unwrap()
        .decode()
        .unwrap();

    let img = img.resize_exact(100, 100, Triangle);

    let pixels = img
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
        .collect();

    while trainer.train_step(&pixels) {
        tx.send(trainer.get_model_params()).unwrap()
    }
}
