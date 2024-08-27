use std::ops::*;


use matrix_operations::Matrix;

use crate::dual::Dual;

struct Arquitecture {
    input_size: usize,
    layer_sizes: Vec<usize>,
}

struct Layer<const K: usize> {
    weights: Matrix<Dual<K>>,
    biases: Matrix<Dual<K>>,
}

struct Mapper {}

impl<const K: usize> Layer<K> {
    fn ingest(&self, inputs: Matrix<Dual<K>>) -> Matrix<Dual<K>> {
        (inputs * self.weights + self.biases).m
    }
}

struct Perceptron<const K: usize> {
    layers: Vec<Layer<K>>,
}

impl<const K: usize> Perceptron<K> {
    fn predict(&self, input: Matrix<Dual<K>>) -> Matrix<Dual<K>> {

        let mut current_layer_activations = input;

        for layer in self.layers {
            current_layer_activations = layer.ingest(current_layer_activations);
        }

        current_layer_activations
    }
}
