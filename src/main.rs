#![recursion_limit = "128"]

use std::vec;

use dual::Dual;
use linear::Linear;

mod dual;
mod linear;

struct DataPoint<const P: usize> {
    input: Vec<Dual<P>>,
    output: Vec<Dual<P>>,
}

impl<const P: usize> DataPoint<P> {
    fn cost(&self, prediction: Vec<Dual<P>>) -> Dual<P> {
        assert_eq!(self.output.len(), prediction.len());

        let mut ret = Dual::cero();

        for (pred_val, goal_val) in prediction.iter().zip(self.output.iter()) {
            let diff = *pred_val - goal_val;
            let diff_sqr = diff * &diff;
            ret += &diff_sqr;
        }

        ret
    }
}

trait Trainable<const P: usize> {
    fn get_params(&self) -> [f32; P];
    fn set_params(&mut self, params: [f32; P]);
    fn eval(&self, input: &Vec<Dual<P>>) -> Vec<Dual<P>>;
}

fn train_step<const P: usize, M: Trainable<P>>(model: &mut M, dataset: &Vec<DataPoint<P>>, learning_factor: f32, learning_max_speed: f32) {
    let mut cost = Dual::cero();

    for data_point in dataset.iter() {
        let prediction = model.eval(&data_point.input);        
        let case_cost = data_point.cost(prediction);

        cost += &case_cost;
    }


    let og_parameters = model.get_params();
    let gradient = cost.get_gradient();

    let mut new_parameters = [0.; P];

    for i in 0..P {
        new_parameters[i] = og_parameters[i] - (learning_factor * gradient[i] / P as f32).max(-learning_max_speed).min(learning_max_speed);
    }

    model.set_params(new_parameters);
}

fn main() {

    let dataset = (0..5).map(|x| {
        DataPoint{
            input: vec![Dual::<2>::new(x as f32)],
            output: vec![Dual::<2>::new(x  as f32 * 3. + 1.)],
        }
    }).collect::<Vec<_>>();

    let mut model = Linear::default();

    for epoch in 1..100{
        train_step(&mut model, &dataset, 5./(epoch+10) as f32, 1.) ;
        println!("#{epoch} model: {:#?}", model);
    }
}
