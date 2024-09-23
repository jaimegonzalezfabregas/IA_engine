use std::array;
use std::ops::{Add, Div, Sub};

use crate::dual::extended_arithmetic::ExtendedArithmetic;
use crate::dual::Dual;
use crate::simd_arr::SimdArr;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const P: usize, const I: usize, const O: usize> {
    pub input: [f32; I],
    pub output: [f32; O],
}

fn datapoint_cost<
    const P: usize,
    const I: usize,
    const O: usize,
    N: ExtendedArithmetic + Clone + Sub<f32, Output = N> + Add<N, Output = N> + Debug + From<f32>,
>(
    goal: &DataPoint<P, I, O>,
    prediction: [N; O],
) -> N {
    let mut ret = N::from(0.);

    for (pred_val, goal_val) in prediction.clone().into_iter().zip(goal.output.into_iter()) {
        let cost = pred_val.clone() - goal_val;
        // println!("    scalar cost for {pred_val:?} and {goal_val:?} is {cost:?}");

        ret = ret + cost.abs();
        // ret = ret + cost.pow2();
    }
    ret
}

fn dataset_cost<
    const P: usize,
    const I: usize,
    const O: usize,
    ExtraData: Sync + Clone,
    N: ExtendedArithmetic
        + Clone
        + Sub<f32, Output = N>
        + Add<f32, Output = N>
        + Debug
        + From<f32>
        + Add<N, Output = N>
        + Div<f32, Output = N>
        + Send
        + Sync,
    F: Fn(&[N; P], &[f32; I], &ExtraData) -> [N; O] + Sync,
>(
    dataset: &Vec<DataPoint<P, I, O>>,
    params: &[N; P],
    model: F,
    extra: &ExtraData,
) -> N {
    let mut accumulator = N::from(0.);
    let cost_list = dataset
        .iter()
        // .progress_count(dataset.len() as u64)
        .map(|data_point| {
            let prediction = (model)(&params, &data_point.input, &extra);

            datapoint_cost(data_point, prediction)
        })
        .collect::<Vec<_>>();


    for cost in cost_list {
        accumulator = accumulator + cost;
    }

    accumulator = accumulator / dataset.len() as f32;

    accumulator
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
    FG: Fn(&[Dual<P, S>; P], &[f32; I], &ExtraData) -> [Dual<P, S>; O] + Sync,
    F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync,
    ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
> {
    model_gradient: FG,
    model: F,
    params: [Dual<P, S>; P],
    param_translator: ParamTranslate,
    extra_data: ExtraData,
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        ExtraData: Sync + Clone,
        S: SimdArr<P>,
        FG: Fn(&[Dual<P, S>; P], &[f32; I], &ExtraData) -> [Dual<P, S>; O] + Sync,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
    > Trainer<P, I, O, ExtraData, S, FG, F, ParamTranslate>
{
    pub fn get_model_params(&self) -> [f32; P] {
        self.params.clone().map(|e| e.get_real())
    }

    pub fn new(
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
    ) -> Self {
        rayon::ThreadPoolBuilder::new()
            .stack_size(1 * 1024 * 1024 * 1024)
            .build_global()
            .unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            model_gradient: trainable_gradient,
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen(), i)),
            param_translator,
            extra_data,
        }
    }

    // TODO return a proper error when NaN apears

    pub fn train_step(&mut self, dataset: &Vec<DataPoint<P, I, O>>) -> bool {
        let cost: Dual<P, S> = dataset_cost(
            dataset,
            &self.params,
            &self.model_gradient,
            &self.extra_data,
        );

        let unit_gradient = cost.get_gradient();
        let og_parameters = array::from_fn(|i| self.params[i].get_real());

        let mut factor = 2.;

        while {
            let gradient = unit_gradient.map(|e| -e * factor);

            let new_params = (self.param_translator)(&og_parameters, &gradient);

            println!("  gradient is : {gradient:?}, took from {og_parameters:?} to {new_params:?}");

            let new_cost: f32 = dataset_cost(dataset, &new_params, &self.model, &self.extra_data);

            println!("  new cost: {}, old_cost: {}", new_cost, cost.get_real());

            if new_cost > cost.get_real() {
                true
            } else {
                // println!("commiting step");

                for (i, param) in new_params.iter().enumerate() {
                    self.params[i].set_real(*param);
                }
                false
            }
        } {
            factor /= 2.;

            if factor < 0.001 {
                return false;
            }
        }
        return true;
    }

    // TODO

    pub fn get_last_cost(&self) -> Option<f32> {
        None
    }
}
