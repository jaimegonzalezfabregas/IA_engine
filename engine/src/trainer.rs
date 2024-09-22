use std::array;

use crate::dual::Dual;
use crate::simd_arr::SimdArr;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const P: usize, const I: usize, const O: usize> {
    pub input: [f32; I],
    pub output: [f32; O],
}

impl<const P: usize, const I: usize, const O: usize> DataPoint<P, I, O> {
    fn cost<S: SimdArr<P>>(&self, prediction: [Dual<P, S>; O]) -> Dual<P, S> {
        let mut ret = Dual::zero();

        for (mut pred_val, goal_val) in prediction.into_iter().zip(self.output.iter()) {
            pred_val = pred_val - *goal_val;

            ret = ret + pred_val.abs();
        }

        ret
    }
}

pub fn default_extra_cost<const P: usize, S: SimdArr<P>>(_: &[Dual<P, S>; P]) -> Dual<P, S> {
    Dual::zero()
}

pub fn default_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    array::from_fn(|i| params[i] + vector[i])
}

pub struct Trainer<
    const P: usize,
    const I: usize,
    const O: usize,
    ExtraData: Sync + Clone,
    S: SimdArr<P>,
    F: Fn(&[Dual<P, S>; P], &[f32; I], &ExtraData) -> [Dual<P, S>; O] + Sync,
    ExtraCost: Fn(&[Dual<P, S>; P]) -> Dual<P, S>,
    ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
> {
    model: F,
    params: [Dual<P, S>; P],
    param_translator: ParamTranslate,
    extra_cost: ExtraCost,
    extra_data: ExtraData,
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        ExtraData: Sync + Clone,
        S: SimdArr<P>,
        F: Fn(&[Dual<P, S>; P], &[f32; I], &ExtraData) -> [Dual<P, S>; O] + Sync,
        ExtraCost: Fn(&[Dual<P, S>; P]) -> Dual<P, S>,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
    > Trainer<P, I, O, ExtraData, S, F, ExtraCost, ParamTranslate>
{
    pub fn get_model_params(&self) -> [f32; P] {
        self.params.clone().map(|e| e.get_real())
    }

    fn cost(&self, dataset: &Vec<DataPoint<P, I, O>>) -> Dual<P, S> {
        let params = &self.params;
        let model = &self.model;
        let extra = self.extra_data.clone();

        let mut accumulator = (&self.extra_cost)(&params);
        let cost_list = dataset
            .par_iter()
            // .progress_count(dataset.len() as u64)
            .map(|data_point| {
                let prediction = (model)(&params, &data_point.input, &extra);

                data_point.cost(prediction)
            })
            .collect::<Vec<_>>();

        for cost in cost_list {
            accumulator = accumulator + cost;
        }

        accumulator / dataset.len() as f32
    }

    pub fn new(
        trainable: F,
        param_translator: ParamTranslate,
        extra_cost: ExtraCost,
        extra_data: ExtraData,
    ) -> Self {
        rayon::ThreadPoolBuilder::new()
            .stack_size(1 * 1024 * 1024 * 1024)
            .build_global()
            .unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen(), i)),
            param_translator,
            extra_cost,
            extra_data,
        }
    }

    // TODO return a proper error when NaN apears

    pub fn train_step(&mut self, dataset: &Vec<DataPoint<P, I, O>>) -> bool {
        let cost = self.cost(dataset);

        let unit_gradient = cost.get_gradient();
        let og_parameters = array::from_fn(|i| self.params[i].get_real());

        // assert_eq!(
        //     false,
        //     unit_gradient
        //         .iter()
        //         .fold(false, |acc, x| acc || !x.is_finite())
        // );

        let mut factor = 2.;

        while {
            let gradient = unit_gradient.map(|e| -e * factor);

            let new_params = (self.param_translator)(&og_parameters, &gradient);

            for (i, param) in new_params.iter().enumerate() {
                self.params[i].set_real(*param);
            }

            self.cost(dataset) > cost
        } {
            factor /= 2.;

            if factor < 0.0001 {
                return false;
            }
        }
        return true;
    }

    pub fn get_last_cost(&self) -> Option<f32> {
        None
    }
}
