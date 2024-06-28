mod perceptron;
use perceptron::{Perceptron, Step};

type Float = f32;

struct DataPoint {
    input: Vec<Float>,
    output: Vec<Float>,
}

fn main() {
    let mut perceptron = Perceptron::new(1).add_layer(4).add_layer(4).add_layer(4).add_layer(4).add_layer(4).add_layer(1);

    let dataset = vec![
        DataPoint {
            input: vec![0.],
            output: vec![0.],
        },
        DataPoint {
            input: vec![0.5],
            output: vec![0.75],
        },
        DataPoint {
            input: vec![1.],
            output: vec![1.],
        },
    ];

    perceptron.train::<true>(&dataset, 1000000, 0.001);

    perceptron.verify::<true>(&dataset);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverter() {
        let mut perceptron = Perceptron::new(1).add_layer(1);

        let dataset = vec![
            DataPoint {
                input: vec![1.],
                output: vec![0.],
            },
            DataPoint {
                input: vec![0.],
                output: vec![1.],
            },
        ];

        perceptron.train::<false>(&dataset, 100, 1.);

        assert!(
            perceptron.verify::<false>(&dataset) < 0.01,
            "verified {}",
            perceptron.verify::<false>(&dataset)
        );
    }

   
}
