use std::array;

use crate::dual::Dual;
use rand::Rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const P: usize, const I: usize, const O: usize> {
    pub input: [f32; I],
    pub output: [f32; O],
}

impl<const P: usize, const I: usize, const O: usize> DataPoint<P, I, O> {
    fn cost(&self, prediction: [Dual<P>; O]) -> Dual<P> {
        let mut ret = Dual::cero();

        for (pred_val, goal_val) in prediction.iter().zip(self.output.iter()) {
            let diff = *pred_val - Dual::new(*goal_val);
            let diff_sqr = diff * diff;
            ret += diff_sqr / 2.;
        }

        ret
    }
}

pub struct Trainer<
    const P: usize,
    const I: usize,
    const O: usize,
    E: Sync + Clone,
    F: Fn(&[Dual<P>; P], &[Dual<P>; I], &E) -> [Dual<P>; O],
    G: Fn(&[Dual<P>; P]) -> [Dual<P>; P],
> {
    model: F,
    params: [Dual<P>; P],
    last_cost: Option<Dual<P>>,
    learning_factor: f32,
    param_transformer: G,
    extra: E
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        E: Sync + Clone,
        F: Fn(&[Dual<P>; P], &[Dual<P>; I], &E) -> [Dual<P>; O] + Sync,
        G: Fn(&[Dual<P>; P]) -> [Dual<P>; P],
    > Trainer<P, I, O, E, F, G>
{
    pub fn get_model_params(&self) -> [f32; P] {
        self.params.map(|e| e.get_real())
    }

    pub fn get_last_cost(&self) -> Option<f32> {
        self.last_cost.map(|e| e.get_real())
    }

    fn cost(&self, dataset: &Vec<DataPoint<P, I, O>>) -> Dual<P> {
        let params = &self.params;
        let model = &self.model;
        let extra = self.extra.clone();

        dataset
            .par_iter()
            // .progress_count(dataset.len() as u64)
            .map(|data_point| {
                let prediction = (model)(&params, &data_point.input.map(|e| Dual::new(e)), &extra);

                let case_cost = data_point.cost(prediction);

                case_cost / dataset.len() as f32
            })
            .reduce(|| Dual::cero(), |acc, cost| acc + cost)
    }

    pub fn new(trainable: F, param_transformer: G, extra: E) -> Self {
        rayon::ThreadPoolBuilder::new()
            .stack_size(1 * 1024 * 1024 * 1024)
            .build_global()
            .unwrap();

        let mut rng = rand::thread_rng();


        Self {
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen(), i)),
            last_cost: None,
            learning_factor: 1.,
            param_transformer,
            extra
        }
    }

    // TODO return a proper error when NaN apears

    pub fn train_step(&mut self, dataset: &Vec<DataPoint<P, I, O>>) -> bool {
        let cost = match self.last_cost {
            None => self.cost(dataset),
            Some(c) => c,
        };
        self.last_cost = Some(cost);

        let unit_gradient = self.last_cost.unwrap().get_gradient();
        let og_parameters = self.params.map(|e| e.get_real());

        //  println!("unit_gradient: {:?}", unit_gradient);

        assert_eq!(false, unit_gradient.iter().fold(false, |acc, x| acc || !x.is_finite()));

        let cost_to_beat = cost.get_real();

        let mut learning_factor = self.learning_factor;

        while self.last_cost.unwrap().get_real() >= cost_to_beat {
            self.params = (self.param_transformer)(&array::from_fn(|i| {
                Dual::new_param(og_parameters[i] - unit_gradient[i] * learning_factor, i)
            }));
            self.last_cost = Some(self.cost(dataset));
            // println!(
            //     "    improvement: {learning_factor} {}",
            //     self.last_cost.unwrap().get_real()
            // );

            learning_factor /= 2.;

            if learning_factor.abs() < 0.00001 {
                self.learning_factor = 1.0;
                // self.last_cost = None;
                return false;
            }
        }

        self.learning_factor = learning_factor * 2.;
        // println!("self.learning_factor {}", self.learning_factor);

        return true;
    }
}
