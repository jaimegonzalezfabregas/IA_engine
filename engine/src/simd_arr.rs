use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

pub mod dense_simd;
pub mod hybrid_simd;
pub mod sparse_simd;

pub trait SimdArr<const S: usize>:
    Debug
    + Sized
    + Send
    + Sync
    + Index<usize, Output = f32>
    + IndexMut<usize, Output = f32>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'b> Sub<&'b Self, Output = Self>
    + Clone
where
    for<'own> &'own Self: DereferenceArithmetic<Self>,
{
    fn new_from_array(seed: &[f32; S]) -> Self;

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self;

    fn zero() -> Self;

    fn neg(&mut self);

    fn to_array(&self) -> [f32; S];
}

pub trait DereferenceArithmetic<In>:
    for<'a> Add<&'a In, Output = In>
    + for<'b> Sub<&'b In, Output = In>
    + Mul<f32, Output = In>
    + Div<f32, Output = In>
{
}
