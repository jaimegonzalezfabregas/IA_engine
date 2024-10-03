use std::fmt::Debug;
use std::ops::{Add, Mul};

use crate::matrix::Matrix;

fn relu<N: PartialOrd<f32> + From<f32>>(x: &mut N) {
    if *x < 0. {
        *x = N::from(0.)
    }
}

pub fn neuronal_network_784_x_10_relu<const X: usize,
    N: Clone
        + Debug
        + From<f32>
        + PartialOrd<f32>
        + PartialOrd<N>
        + Add<N, Output = N>
        + Mul<N, Output = N>
        + Mul<f32, Output = N>,
>(
    params: &[N; 800 * 784 + 800 + 10 * 800 + 10],
    input: &[f32; 784],
    _: &(),
) -> [N; 10] {
    let input: Matrix<784, 1, _> = Matrix::from_flat(&input.map(N::from));

    let hidden_layer_weights: Matrix<X, 784, N> = Matrix::from_flat(&params[0..(X * 784)]);
    let hidden_layer_biases: Matrix<X, 1, N> =
        Matrix::from_flat(&params[(X * 784)..(X * 785)]);

    let excitations_layer1 = hidden_layer_weights * input;
    let mut activations_layer1 = hidden_layer_biases + excitations_layer1;

    activations_layer1.delinearize(relu);

    let hidden_layer_weights: Matrix<10, X, N> =
        Matrix::from_flat(&params[(X * 785)..(X * 785 + 10 * X)]);

    let hidden_layer_biases: Matrix<10, 1, N> =
        Matrix::from_flat(&params[(X * 785 + 10 * X)..]);

    let excitations_layer2 = hidden_layer_weights * activations_layer1;
    let mut out = hidden_layer_biases + excitations_layer2;

    out.delinearize(relu);

    out.as_array()
}
