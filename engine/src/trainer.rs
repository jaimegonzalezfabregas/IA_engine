use std::array;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::ops::{Add, Div, Sub};

use crate::dual::extended_arithmetic::ExtendedArithmetic;
use crate::dual::Dual;
use crate::simd_arr::dense_simd::DenseSimd;
use crate::simd_arr::hybrid_simd::HybridSimd;
use crate::simd_arr::SimdArr;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const P: usize, const I: usize, const O: usize> {
    pub input: [f32; I],
    pub output: [f32; O],
}

pub struct CriticalityCue<const CRITICALITY: usize>();

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
        // println!("{ret:?} + {cost:?}");

        ret = ret + cost.abs();

        // ret = ret + cost.pow2();
    }
    ret
}

fn dataset_cost<
    'a,
    'b,
    const PARALELIZE: bool,
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
    D: IntoIterator<Item = &'b DataPoint<P, I, O>>
        + IntoParallelIterator<Item = &'a DataPoint<P, I, O>>,
>(
    dataset: D,
    dataset_len: usize,
    params: &[N; P],
    model: F,
    extra: &ExtraData,
) -> N {
    let mut accumulator = N::from(0.);
    let cost_list = if PARALELIZE {
        dataset
            .into_par_iter()
            // .progress_count(dataset_len as u64)
            .map(|data_point| {
                let prediction = (model)(&params, &data_point.input, &extra);

                datapoint_cost(&data_point, prediction)
            })
            .collect::<Vec<_>>()
    } else {
        dataset
            .into_iter()
            .map(|data_point| {
                let prediction = (model)(&params, &data_point.input, &extra);

                datapoint_cost(&data_point, prediction)
            })
            .collect::<Vec<_>>()
    };

    for cost in cost_list {
        accumulator = accumulator + cost;
    }

    accumulator = accumulator / dataset_len as f32;

    accumulator
}

pub fn default_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    array::from_fn(|i| params[i] + vector[i])
}

pub fn param_translator_with_bounds<const P: usize, const MAX: isize, const MIN: isize>(
    params: &[f32; P],
    vector: &[f32; P],
) -> [f32; P] {
    array::from_fn(|i| (params[i] + vector[i]).min(MAX as f32).max(MIN as f32))
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
    last_cost: Option<f32>,
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        ExtraData: Sync + Clone,
        FG: Fn(&[Dual<P, DenseSimd<P>>; P], &[f32; I], &ExtraData) -> [Dual<P, DenseSimd<P>>; O] + Sync,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
    > Trainer<P, I, O, ExtraData, DenseSimd<P>, FG, F, ParamTranslate>
{
    pub fn new_dense(
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
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            param_translator,
            extra_data,
            last_cost: None,
        }
    }
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        const CRITIALITY: usize,
        ExtraData: Sync + Clone,
        FG: Fn(
                &[Dual<P, HybridSimd<P, CRITIALITY>>; P],
                &[f32; I],
                &ExtraData,
            ) -> [Dual<P, HybridSimd<P, CRITIALITY>>; O]
            + Sync,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P],
    > Trainer<P, I, O, ExtraData, HybridSimd<P, CRITIALITY>, FG, F, ParamTranslate>
{
    pub fn new_hybrid(
        _: CriticalityCue<CRITIALITY>,
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
    ) -> Self {
        rayon::ThreadPoolBuilder::new()
            .stack_size(1 * 1024 * 1024 * 1024)
            .build_global();

        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            model_gradient: trainable_gradient,
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            param_translator,
            extra_data,
            last_cost: None,
        }
    }
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

    pub fn save(&self, file_path: &str) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)?;

        for p in self.params.iter() {
            file.write(format!("{}\n", p.get_real()).as_bytes())?;
        }

        Ok(())
    }

    pub fn load(&mut self, file_path: &str) -> std::io::Result<()> {
        let file = OpenOptions::new().read(true).open(file_path)?;
        let reader = io::BufReader::new(file);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if let Ok(param) = line.parse::<f32>() {
                self.params[i].set_real(param);
            } else {
                eprintln!("Failed to parse line: {}", line);
            }
        }

        Ok(())
    }

    pub fn shake(&mut self, factor: f32) {
        for i in 0..P {
            self.params[i]
                .set_real(self.params[i].get_real() + (rand::random::<f32>() - 0.5) * factor);
        }
    }

    // TODO partition the dataset

    pub fn train_stocastic_step<const PARALELIZE: bool, CB: Fn(usize, &Self)>(
        &mut self,
        dataset: &Vec<DataPoint<P, I, O>>,
        subdataset_size: usize,
        inter_step_callback: CB,
    ) -> bool {
        let mut ret = false;
        for (i, sub_dataset) in dataset.chunks(subdataset_size).enumerate() {
            ret |= self.train_step::<PARALELIZE, _>(sub_dataset, sub_dataset.len());
            inter_step_callback(i, &self);
        }

        ret
    }

    // TODO adam

    // TODO return a proper error when NaN apears

    pub fn train_step<
        'a,
        'b,
        const PARALELIZE: bool,
        D: IntoIterator<Item = &'b DataPoint<P, I, O>>
            + IntoParallelIterator<Item = &'a DataPoint<P, I, O>>
            + Clone,
    >(
        &mut self,
        dataset: D,
        dataset_len: usize,
    ) -> bool {
        let cost: Dual<P, S> = dataset_cost::<PARALELIZE, _, _, _, _, _, _, _>(
            dataset.clone(),
            dataset_len,
            &self.params,
            &self.model_gradient,
            &self.extra_data,
        );

        let mut factor = 1.;

        let raw_gradient = cost.get_gradient();
        let gradient_size = raw_gradient.iter().fold(0., |acc, elm| acc + (elm * elm));

        let unit_gradient = array::from_fn(|i| raw_gradient[i] / gradient_size.sqrt());
        let og_parameters = array::from_fn(|i| self.params[i].get_real());

        // println!("  raw gradient is : {raw_gradient:?}");

        while {
            let gradient = unit_gradient.map(|e| -e * factor);

            let new_params = (self.param_translator)(&og_parameters, &gradient);

            for (i, param) in new_params.iter().enumerate() {
                self.params[i].set_real(*param);
            }

            let new_cost: f32 = dataset_cost::<PARALELIZE, _, _, _, _, _, _, _>(
                dataset.clone(),
                dataset_len,
                &new_params,
                &self.model,
                &self.extra_data,
            );
            self.last_cost = Some(new_cost);

            new_cost > cost.get_real()
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
        self.last_cost
    }

    pub fn eval(&self, input: &[f32; I]) -> [f32; O] {
        (self.model)(
            &self.params.clone().map(|e| e.get_real()),
            input,
            &self.extra_data,
        )
    }
}
