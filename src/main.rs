mod perceptron;
use perceptron::Perceptron;

type Float = f32;

struct DataPoint {
    input: Vec<Float>,
    output: Vec<Float>,
}

fn main() {
    let perceptron = Perceptron::new(2).add_layer(2).add_layer(2).add_layer(1);
}

#[cfg(test)]
mod tests {
    use crate::{perceptron::{Perceptron, Step}, DataPoint};

    #[test]
    fn simple_train() {
        let perceptron = Perceptron::new(1).add_layer(1);

        let cases = vec![
            DataPoint {
                input: vec![1.],
                output: vec![0.],
            },
            DataPoint {
                input: vec![0.],
                output: vec![1.],
            },
        ];

        for _ in 0..100 {
            let steps = cases.iter().map(|c| perceptron.back_propagation(&c.input, &c.output)).collect();

            let step = Step::add(steps);

            perceptron.apply(step);
        }
    }
}
