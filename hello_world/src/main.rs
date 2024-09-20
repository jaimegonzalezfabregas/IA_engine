#![feature(generic_arg_infer)]

use ia_engine::{simd_arr::dense_simd::DenseSimd, trainer::{default_extra_cost, default_param_translator, DataPoint, Trainer}};

fn direct<N: Clone>(parameters: &[N; 1], _: &[N; 0], _: &()) -> [N; 1] {
    parameters.clone()
}

fn main() {
    let dataset = vec![DataPoint {
        input: [],
        output: [200.],
    }];

    let mut trainer: Trainer<_, _, _, _, DenseSimd<_>, _, _, _> =
        Trainer::new(direct, default_param_translator, default_extra_cost, ());

    for _ in 0..1000 {
        trainer.train_step(&dataset);
        println!("{:?}", trainer.get_model_params());
    }
}
