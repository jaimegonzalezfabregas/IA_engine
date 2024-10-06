#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

mod matrix;
mod mnist;
mod neuronal_network;

use std::thread;

use ia_engine::trainer::{
    default_param_translator, param_translator_with_bounds, CriticalityCue, DataPoint, Trainer,
};
use mnist::load_data;
use neuronal_network::neuronal_network;

fn main() {
    thread::Builder::new()
        .stack_size(20_000_000_000)
        .name("big stack main".into())
        .spawn(big_stack_main)
        .unwrap()
        .join()
        .unwrap();
}

fn big_stack_main() {
    println!("start!");
    let dataset = load_data("mnist/t10k").unwrap();
    // let dataset = vec![
    //     DataPoint {
    //         input: [0., 0.],
    //         output: [0.],
    //     },
    //     DataPoint {
    //         input: [0., 1.],
    //         output: [1.],
    //     },
    //     DataPoint {
    //         input: [1., 0.],
    //         output: [1.],
    //     },
    //     DataPoint {
    //         input: [1., 1.],
    //         output: [0.],
    //     },
    // ];

    let mut trainer = Trainer::new_heap_hybrid(
        CriticalityCue::<{63610 / 4}>(),
        neuronal_network::<{ 28 * 28 }, 10, 63610, _>,
        neuronal_network::<{ 28 * 28 }, 10, _, _>,
        default_param_translator,
        // param_translator_with_bounds::<_, 4, -4>,
        vec![28*28, 80, 10],
    );

    while trainer.train_step::<true>(&dataset) {
        // println!("{:?}", trainer.get_model_params());
        println!("{:?}", trainer.get_last_cost());
    }

    // for dp in dataset {
    //     let prediction = neuronal_network::<{ 28 * 28 }, 10, _, _>(
    //         &trainer.get_model_params(),
    //         &dp.input,
    //         &vec![28*28, 80, 10],
    //     );

    //     println!(
    //         "for input: {:?} got {:?} aproximating {:?}",
    //         dp.input,
    //         prediction[0],
    //         dp.output[0],
    //     );
    // }
}
