#![feature(generic_arg_infer)]

use ia_engine::{
    simd_arr::dense_simd::DenseSimd,
    trainer::{default_param_translator, DataPoint, Trainer},
};

fn direct<N: Clone>(parameters: &[N; 1], _: &[f32; 0], _: &()) -> [N; 1] {
    parameters.clone()
}

fn main() {
    let dataset = vec![DataPoint {
        input: [],
        output: [-200.],
    }];

    let mut trainer: Trainer<_, _, _, _, DenseSimd<_>, _, _, _> = Trainer::new(
        direct,
        direct,
        default_param_translator,
        (),
    );

    while trainer.train_step(&dataset) {
        println!("{:?}", trainer.get_model_params());
    }
    println!("{:?}", trainer.get_model_params());
}
