use std::{array, sync::mpsc::Sender};

use ia_engine::trainer::{CriticalityCue, Trainer};

use crate::{
    dataset_sample_service::DatasetSampleService, tiler, TrainerComunicationCodes, TILE_COUNT,
    TILE_COUNT_SQRT,
};

fn max_speed_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    array::from_fn(|i| (params[i] + vector[i]).max(0.).min(1.))
}

pub fn train_thread(
    tx: Sender<TrainerComunicationCodes<([f32; TILE_COUNT * 5], (Option<f32>, usize))>>,
    max_iterations: Option<usize>,
) {
    train_work(Some(tx), max_iterations);
    // train_work::<DenseSimd<_>>(Some(tx), max_iterations);
}

pub(crate) fn train_work(
    tx: Option<Sender<TrainerComunicationCodes<([f32; TILE_COUNT * 5], (Option<f32>, usize))>>>,
    max_iterations: Option<usize>,
) {
    let mut trainer = Trainer::new_hybrid(
        CriticalityCue::<{ TILE_COUNT / 2 }>(),
        tiler,
        tiler,
        max_speed_param_translator,
        (),
    );

    if let Some(ref tx) = tx {
        tx.send(TrainerComunicationCodes::Msg((
            trainer.get_model_params(),
            (None, 0),
        )))
        .unwrap();
    }

    let dataset_service = DatasetSampleService::new(
        "assets/rust.png",
        TILE_COUNT_SQRT as f32 / (TILE_COUNT_SQRT - 2) as f32,
    );

    let mut local_minimum_count = 0;
    let mut iterations = 0;
    while local_minimum_count < 100 {
        let pixels = dataset_service.get(100, &trainer.get_model_params());

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

        if !trainer.train_step::<true, _>(&pixels, pixels.len()) {
            local_minimum_count += 1;
            trainer.shake(0.1);
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
