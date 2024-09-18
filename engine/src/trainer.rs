use std::array;
use std::ops::{Add, Div, Mul, Sub};

use crate::dual::Dual;
use crate::simd_arr::{DereferenceArithmetic, SimdArr};
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
    fn cost<S: SimdArr<P>>(&self, prediction: [Dual<P, S>; O]) -> Dual<P, S>
    where
        for<'own> &'own S: DereferenceArithmetic<S>,
    {
        let mut ret = Dual::zero();

        for (pred_val, goal_val) in prediction.iter().zip(self.output.iter()) {
            let mut diff = pred_val - &Dual::new(*goal_val);
            diff.abs();
            ret += &diff / 2.;
        }

        ret
    }
}

pub fn default_extra_cost<const P: usize, S: SimdArr<P>>(_: &[Dual<P, S>; P]) -> Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
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
    F: Fn(&[Dual<P, S>; P], &[Dual<P, S>; I], &ExtraData) -> [Dual<P, S>; O],
    ExtraCost: Fn(&[Dual<P, S>; P]) -> Dual<P, S>,
    ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
> where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    model: F,
    params: [Dual<P, S>; P],
    param_translator: ParamTranslate,
    extra_cost: ExtraCost,
    extra_data: ExtraData,
}

// type SimdImplementation<const P: usize> = SparseSimd<P>;
//    93.31s user 2.76s system 329% cpu 29.126 total
//    119.51s user 3.30s system 366% cpu 33.478 total
// type SimdImplementation<const P: usize> = DenseSimd<P>;
//    69.93s user 2.74s system 389% cpu 18.672 total
//    111.22s user 5.09s system 356% cpu 32.620 total
// type SimdImplementation<const P: usize> = HybridSimd<P>;
//    137.31s user 3.22s system 380% cpu 36.937 total
//    144.43s user 3.13s system 396% cpu 37.252 total

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        ExtraData: Sync + Clone,
        S: SimdArr<P>,
        F: Fn(&[Dual<P, S>; P], &[Dual<P, S>; I], &ExtraData) -> [Dual<P, S>; O] + Sync,
        ExtraCost: Fn(&[Dual<P, S>; P]) -> Dual<P, S>,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
    > Trainer<P, I, O, ExtraData, S, F, ExtraCost, ParamTranslate>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    pub fn get_model_params(&self) -> [f32; P] {
        self.params.clone().map(|e| e.get_real())
    }

    fn cost(&self, dataset: &Vec<DataPoint<P, I, O>>) -> Dual<P, S> {
        let params = &self.params;
        let model = &self.model;
        let extra = self.extra_data.clone();

        dataset
            .par_iter()
            // .progress_count(dataset.len() as u64)
            .map(|data_point| {
                let prediction = (model)(&params, &data_point.input.map(|e| Dual::new(e)), &extra);

                let case_cost = data_point.cost(prediction);

                &case_cost / dataset.len() as f32
            })
            .reduce(|| Dual::zero(), |acc, cost| acc + cost)
            + (&self.extra_cost)(&params)
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

        let gradient = unit_gradient.map(|e| -e);

        let new_params = (self.param_translator)(&og_parameters, &gradient);

        for (i, param) in new_params.iter().enumerate() {
            self.params[i].set_real(*param);
        }

        return true;
    }
}
