use std::array;

use crate::dual::Dual;

#[derive(Debug)]
pub struct DataPoint<const P: usize, const I: usize, const O: usize> {
    pub input: [f32; I],
    pub output: [f32; O],
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

pub trait Trainable<const P: usize, const I: usize, const O: usize> {
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

        cost / &Dual::<P>::new(dataset.len() as f32)
    }
}

pub struct Trainer<const P: usize, const I: usize, const O: usize, T: Trainable<P, I, O>> {
    model: T,
    last_cost: Option<Dual<P>>,
    learning_factor: f32,
}

impl<const P: usize, const I: usize, const O: usize, T: Trainable<P, I, O>> Trainer<P, I, O, T> {
    pub fn get_model_params(&self) -> [f32;P]{
        self.model.get_params()
    }
    
    pub fn new(trainable: T) -> Self {
        Self {
            model: trainable,
            last_cost: None,
            learning_factor: 1.,
        }
    }

    pub fn eval(&self, input: [f32; I]) -> [f32; O]{
        self.model.eval(&array::from_fn(|i| Dual::new(input[i]))).map(|e|e.get_real())
    }

    pub fn train_step(&mut self, dataset: &Vec<DataPoint<P, I, O>>) -> bool {
        let cost = match self.last_cost {
            None => self.model.cost(dataset),
            Some(c) => c,
        };
        self.last_cost = Some(cost);

        let unit_gradient = self.last_cost.unwrap().get_gradient();
        let og_parameters = self.model.get_params();

        let cost_to_beat = cost.get_real();
        // println!("cost to beat: {cost_to_beat}");

        let mut learning_factor = self.learning_factor;

        while self.last_cost.unwrap().get_real() >= cost_to_beat {
            self.model.set_params(array::from_fn(|i| {
                og_parameters[i] - unit_gradient[i] * learning_factor
            }));
            self.last_cost = Some(self.model.cost(dataset));
            // println!(
            //     "    improvement: {learning_factor} {}",
            //     self.last_cost.unwrap().get_real()
            // );

            learning_factor /= 2.;

            if learning_factor.abs() < 0.000000000000001 {
                self.learning_factor = 10000.;
                return false;
            }
        }

        self.learning_factor = learning_factor * 8.;

        return true;
    }
}