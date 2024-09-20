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
    + Clone
{
    fn new_from_array(seed: &[f32; S]) -> Self;

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self;

    fn zero() -> Self;

    fn neg(&mut self);

    fn to_array(&self) -> [f32; S];

    fn acumulate<RHS: SimdArr<S>>(&mut self, rhs: &RHS);

    fn multiply(&mut self, rhs: f32);
}