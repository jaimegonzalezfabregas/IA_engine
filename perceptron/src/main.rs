#![feature(generic_arg_infer)]

mod mnist;
mod neuronal_network;

use ia_engine::{
    simd_arr::dense_simd::DenseSimd,
    trainer::{default_param_translator, DataPoint, Trainer},
};
use mnist::load_data;
use neuronal_network::neuronal_network;



fn main() {
    let dataset = load_data("mnist/t10k");

    let mut trainer: Trainer<_, _, _, _, DenseSimd<_>, _, _, _> =
        Trainer::new(neuronal_network, neuronal_network, default_param_translator, ());

    while trainer.train_step(&dataset) {
        println!("{:?}", trainer.get_model_params());
    }
    println!("{:?}", trainer.get_model_params());
}
