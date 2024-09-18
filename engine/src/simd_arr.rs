use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};

pub mod dense_simd;
pub mod hybrid_simd;
pub mod sparse_simd;

pub trait SimdArr<const S: usize>:
    Debug
    + Send
    + Sync
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + Index<usize, Output = f32>
    + IndexMut<usize, Output = f32>
where
    for<'a, 'b> Self: Add<&'a Self, Output = Self> + Sub<&'b Self, Output = Self>,
    for<'a, 'b, 'c> &'c Self: Add<&'a Self, Output = Self> + Sub<&'b Self, Output = Self>,
{
    fn new_from_array(seed: &[f32; S]) -> Self;

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self;

    fn zero() -> Self;

    fn to_array(&self) -> [f32; S];
}
