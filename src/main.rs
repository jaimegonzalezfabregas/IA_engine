mod perceptron;
use perceptron::{Perceptron, Step};

type Float = f32;

struct DataPoint {
    input: Vec<Float>,
    output: Vec<Float>,
}

fn main() {
    let mut perceptron = Perceptron::new(1).add_layer(1);

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
        let steps = cases
            .iter()
            .map(|c| perceptron.back_propagation(&c.input, &c.output))
            .collect();

        let step = Step::add(steps, perceptron.get_empty_step());

        perceptron.apply(step);
    }

    println!("{:?}", perceptron.predict(&vec![1.]));
    println!("{:?}", perceptron.predict(&vec![0.]));
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    #[test]
    fn inverter() {
        let mut perceptron = Perceptron::new(1).add_layer(1);

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
            let steps = cases
                .iter()
                .map(|c| perceptron.back_propagation(&c.input, &c.output))
                .collect();

            let step = Step::add(steps, perceptron.get_empty_step());

            perceptron.apply(step);
        }

        assert!(perceptron.predict(&vec![1.])[0].activations[0] < 0.01);
        assert!(perceptron.predict(&vec![0.])[0].activations[0] - 1. < 0.01);
    }

    #[test]
    fn rotator() {
        let mut perceptron = Perceptron::new(3)
            .add_layer(4)
            .add_layer(4)
            .add_layer(4)
            .add_layer(2);

        let cases: Vec<DataPoint> = (0..360)
            .flat_map(|angle| {
                (-10..10).flat_map(move |x| {
                    (-10..10).map(move |y| DataPoint {
                        input: vec![angle as Float, x as Float, y as Float],
                        output: vec![
                            (x as Float) * (angle as Float / 360. * 2. * PI).sin()
                                + (y as Float) * (angle as Float).cos(),
                            (y as Float) * (angle as Float / 360. * 2. * PI).sin()
                                + (x as Float) * (angle as Float).cos(),
                        ],
                    })
                })
            })
            .collect();

        for _ in 0..10 {

            let steps = cases
                .iter()
                .map(|c| perceptron.back_propagation(&c.input, &c.output))
                .collect();

            let step = Step::add(steps, perceptron.get_empty_step());

            perceptron.apply(step);
        }

        for data_point in cases {
            let prediction = perceptron.predict(&data_point.input);
            let output = &prediction.last().unwrap().activations;

            for (a, b) in output.iter().zip(data_point.output) {
                assert!(a - b < 0.01);
            }
        }

        assert!(perceptron.predict(&vec![0.])[0].activations[0] - 1. < 0.01);
    }
}
