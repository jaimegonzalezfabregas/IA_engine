type Float = f32;

fn random() -> Float {
    1.
}

struct Layer {
    neuron_count: usize,
    previous_layer_neuron_count: usize,
    weights: Vec<Vec<Float>>,
    biases: Vec<Float>,
}

impl Layer {
    fn new(neuron_count: usize, previous_layer_neuron_count: usize) -> Self {
        let weights = (0..neuron_count)
            .map(|_| (0..previous_layer_neuron_count).map(|_| random()).collect())
            .collect();
        let biases = (0..neuron_count).map(|_| random()).collect();

        Layer {
            neuron_count,
            previous_layer_neuron_count,
            weights,
            biases,
        }
    }

    fn activation_fn(&self, input: Float) -> Float {
        input.max(0.)
    }

    fn predict(&self, input: Vec<Float>) -> Vec<Float> {
        (0..self.neuron_count)
            .map(|i| {
                self.activation_fn(
                    input
                        .iter()
                        .enumerate()
                        .fold(self.biases[i], |acc, (j, input)| {
                            acc + input * self.weights[i][j]
                        }),
                )
            })
            .collect()
    }
}

struct Gradient {
    weights_d: Vec<Vec<Float>>,
    biases_d: Vec<Float>,
}

struct Perceptron {
    input_size: usize,
    layers: Vec<Layer>,
}

impl Perceptron {
    pub fn new(input_size: usize) -> Self {
        Perceptron {
            input_size,
            layers: vec![],
        }
    }

    pub fn add_layer(mut self, neuron_count: usize) -> Self {
        let layer = Layer::new(
            neuron_count,
            self.layers
                .last()
                .map(|l| l.neuron_count)
                .unwrap_or(self.input_size),
        );
        self.layers.push(layer);
        self
    }

    fn predict_engine(&self, input: Vec<Float>, layer_i: usize) -> Vec<Float> {
        self.layers[layer_i].predict(if layer_i == 0 {
            input
        } else {
            self.predict_engine(input, layer_i - 1)
        })
    }

    pub fn predict(&self, input: Vec<Float>) -> Vec<Float> {
        self.predict_engine(input, self.layers.len() - 1)
    }
}
