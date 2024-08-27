use std::array;

use dual::Dual;
use linear::Linear;
use polinomial::Polinomial;

mod dual;
mod linear;
mod polinomial;

#[derive(Debug)]
struct DataPoint<const P: usize, const I: usize, const O: usize> {
    input: [f32; I],
    output: [f32; O],
}

impl<const P: usize, const I: usize, const O: usize> DataPoint<P, I, O> {
    fn cost(&self, prediction: [Dual<P>; O]) -> Dual<P> {
        let mut ret = Dual::cero();

        for (pred_val, goal_val) in prediction.iter().zip(self.output.iter()) {
            let diff = *pred_val - &Dual::new(*goal_val);
            let diff_sqr = diff * &diff;
            ret += &diff_sqr;
        }

        ret
    }
}

trait Trainable<const P: usize, const I: usize, const O: usize> {
    fn get_params(&self) -> [f32; P];
    fn set_params(&mut self, params: [f32; P]);
    fn eval(&self, input: &[Dual<P>; I]) -> [Dual<P>; O];
    fn cost(&self, dataset: &Vec<DataPoint<P, I, O>>) -> Dual<P> {
        let mut cost = Dual::cero();

        for data_point in dataset.iter() {
            let prediction = self.eval(&data_point.input.map(|e| Dual::new(e)));
            let case_cost = data_point.cost(prediction);

            cost += &case_cost;
        }

        cost
    }
    fn train_step(
        &mut self,
        dataset: &Vec<DataPoint<P, I, O>>,
        learning_factor: f32,
        learning_max_speed: f32,
    ) {
        let cost = self.cost(dataset);

        let gradient = cost.get_gradient().map(|x| learning_factor * x / P as f32);

        let step_size = gradient.iter().fold(0., |sum, e| sum + e.powf(2.)).sqrt();

        let clamped_gradient = if step_size > learning_max_speed {
            array::from_fn(|i| gradient[i] / step_size * learning_max_speed)
        } else {
            gradient
        };

        // println!("gradient is {gradient:?} {clamped_gradient:?}", );

        let og_parameters = self.get_params();
        let new_parameters = array::from_fn(|i| og_parameters[i] - clamped_gradient[i]);

        self.set_params(new_parameters);
    }
}

fn main() {
    let mut model = Polinomial::<9>::new();

    let dataset = (-10..10)
    .map(|x| x as f32 / 10.)
        .map(|x| DataPoint {
            input: [x],
            output: [
                1. * (x * x * x * x * x* x * x) 
                + 2. * (x * x * x * x * x * x) 
                + 3. * (x * x * x * x * x) 
                + 4. * (x * x * x * x) 
                + 5. * (x * x * x) 
                + 6. * (x * x) 
                + 7. * x
                + 8.,
            ],
        })
        .collect::<Vec<_>>();

    println!("{dataset:?}");

    for epoch in 1.. {
        if epoch % 10000 == 0 {
            println!("model: {:?}", model.get_params());
        }
        model.train_step(&dataset, 1., 0.001);
    }
}
