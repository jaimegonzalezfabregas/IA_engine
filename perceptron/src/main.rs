#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

mod matrix;
mod mnist;
mod neuronal_network;

use std::thread;

use ia_engine::trainer::{default_param_translator, Trainer};
use mnist::load_data;
use neuronal_network::neuronal_network_784_x_10_relu;

fn main() {
    thread::Builder::new()
        .stack_size(30_000_000)
        .spawn(big_stack_main)
        .unwrap()
        .join()
        .unwrap();
}

fn big_stack_main() {
    println!("start!");
    let dataset = load_data("mnist/t10k").unwrap();

    let mut trainer = Trainer::new_hybrid(
        neuronal_network_784_x_10_relu::<1, _>,
        neuronal_network_784_x_10_relu::<1, _>,
        default_param_translator,
        (),
    );

    while trainer.train_step(&dataset) {
        println!("{:?}", trainer.get_model_params());
        println!("{:?}", trainer.get_last_cost());
    }
    println!("{:?}", trainer.get_model_params());
    println!("{:?}", trainer.get_last_cost());
}
