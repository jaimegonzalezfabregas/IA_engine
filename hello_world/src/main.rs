#![feature(generic_arg_infer)]

use ia_engine::trainer::{default_extra_cost, default_param_translator, DataPoint, Trainer};

fn direct<N: Copy>(parameters: &[N; 1], _: &[N; 0], _: &()) -> [N; 1] {
    *parameters
}

fn main() {
    let dataset = vec![DataPoint {
        input: [],
        output: [200.],
    }];

    let mut trainer = Trainer::new(direct, default_param_translator, default_extra_cost, ());

    for _ in 0..10 {
        trainer.train_step(&dataset);
        println!("{:?}", trainer.get_model_params());
    }
}
