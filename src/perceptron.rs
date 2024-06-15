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

struct BackPropStep {
    prev_layer_activation_gradient: Vec<Float>,
    weights_step: Vec<Vec<Float>>,
    biases_step: Vec<Float>,
}
struct LayerState {
    activations: Vec<Float>,
    stimulations: Vec<Float>,
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

    fn activation_fn(&self, input: &Float) -> Float {
        input.max(0.)
    }

    fn activation_fn_d(&self, input: &Float) -> Float {
        if *input > 0. {
            1.
        } else {
            0.
        }
    }

    fn activation(&self, input: &Vec<Float>) -> LayerState {
        let stimulations: Vec<Float> = (0..self.neuron_count)
            .map(|i| {
                input
                    .iter()
                    .enumerate()
                    .fold(self.biases[i], |acc, (j, input)| {
                        acc + input * self.weights[i][j]
                    })
            })
            .collect();

        LayerState {
            activations: stimulations.iter().map(|f| self.activation_fn(f)).collect(),
            stimulations,
        }
    }

    fn back_propagation(
        &self,
        previous_activation: &Vec<Float>,
        current_layer_state: &LayerState,
        current_layer_activation_gradient: &Vec<Float>,
    ) -> BackPropStep {
        let mut ret = BackPropStep {
            prev_layer_activation_gradient: vec![0.; previous_activation.len()],
            biases_step: vec![0.; current_layer_activation_gradient.len()],
            weights_step: vec![
                vec![0.; previous_activation.len()];
                current_layer_activation_gradient.len()
            ],
        };

        for (i, gradient) in current_layer_activation_gradient.iter().enumerate() {
            ret.biases_step[i] +=
                gradient * self.activation_fn_d(&current_layer_state.stimulations[i]);

            for (j, weight) in self.weights[i].iter().enumerate() {
                ret.weights_step[i][j] += gradient
                    * self.activation_fn_d(&current_layer_state.stimulations[i])
                    * previous_activation[j];

                ret.prev_layer_activation_gradient[i] +=
                    gradient * self.activation_fn_d(&current_layer_state.stimulations[i]) * weight;
            }
        }

        ret
    }
}

struct Step {
    weights: Vec<Vec<Vec<Float>>>,
    biases: Vec<Vec<Float>>,
}

pub struct Perceptron {
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

    pub fn predict(&self, input: &Vec<Float>) -> Vec<LayerState> {
        let mut ret = vec![];

        ret[0] = self.layers[0].activation(input);

        for i in 1..self.layers.len() {
            ret[i] = self.layers[i].activation(&ret[i - 1].activations);
        }

        ret
    }

    pub fn back_propagation(&self, input: &Vec<Float>, output: &Vec<Float>) -> Step {
        let mut ret = Step {
            weights: vec![],
            biases: vec![],
        };

        let layer_states = self.predict(&input);
        let mut activation_gradients: Vec<Vec<Float>> = vec![];

        for i in 0..layer_states.len() {
            let layer_i = layer_states.len() - 1 - i;
            let activation_gradient: Vec<Float> = if i == 0 {
                layer_states
                    .last()
                    .unwrap()
                    .activations
                    .iter()
                    .zip(output)
                    .map(|(activation, desired_activation)| desired_activation - activation)
                    .collect()
            } else {
                activation_gradients[i - 1].clone()
            };

            let backprop_step = self.layers[layer_i].back_propagation(
                if layer_i == 0 {
                    &input
                } else {
                    &layer_states[layer_i - 1].activations
                },
                &layer_states[layer_i],
                &activation_gradient,
            );

            activation_gradients[i] = backprop_step.prev_layer_activation_gradient;
            ret.weights[layer_i] = backprop_step.weights_step;
            ret.biases[layer_i] = backprop_step.biases_step;
        }

        ret
    }
}
