use std::array;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul};

use ia_engine::dual::extended_arithmetic::ExtendedArithmetic;

use crate::matrix::Matrix;

fn relu<N: ExtendedArithmetic>(x: &mut N) {
    x.relu_on_mut()
}

fn sigmoid<N: ExtendedArithmetic + Debug>(x: &mut N) {
    x.sigmoid_on_mut();
}

pub fn neuronal_network<
    const I: usize,
    const O: usize,
    const P: usize,
    N: Clone
        + Debug
        + From<f32>
        + PartialOrd<f32>
        + PartialOrd<N>
        + Add<N, Output = N>
        + Mul<N, Output = N>
        + Mul<f32, Output = N>
        + Div<N, Output = N>
        + ExtendedArithmetic,
>(
    params: &[N; P],
    input: &[f32; I],
    structure: &Vec<usize>,
) -> [N; O] {
    {
        assert_eq!(structure[0], I);
        assert_eq!(structure[structure.len() - 1], O);

        let mut spected_parameter_count = 0;

        for i in 0..(structure.len() - 1) {
            spected_parameter_count += (structure[i] + 1) * structure[i + 1]
        }
        assert_eq!(spected_parameter_count, P);
    }

    let mut propagation = Matrix::deserialize(1, structure[0], &input.map(N::from));

    let parameter_cursor = 0;

    for i in 0..(structure.len() - 1) {
        let layer_size = (structure[i] + 1) * structure[i + 1];

        let layer_weights = Matrix::deserialize(
            structure[i] + 1,
            structure[i + 1],
            &params[parameter_cursor..(parameter_cursor + layer_size)],
        );

        // print!(
        //     "{:#?}, {parameter_cursor}, {layer_size}",
        //     &params[parameter_cursor..(parameter_cursor + layer_size)]
        // );

        propagation = layer_weights * propagation.add_bias();

        propagation.delinearize(sigmoid);
    }

    let vec_out = propagation.serialize();

    let exp_out = vec_out.into_iter().map(|x| x.exp()).collect::<Vec<_>>();

    let mut acc = N::from(0.);
    for n in exp_out.iter() {
        acc.accumulate(n);
    }

    let ret = array::from_fn(|i| exp_out[i].clone() / acc.clone());

    ret
}
