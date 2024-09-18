use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use super::{dense_simd::DenseSimd, sparse_simd::SparseSimd, DereferenceArithmetic, SimdArr};

#[derive(Clone, Copy, Debug)]
pub enum HybridSimd<const SIZE: usize, const CRITIALITY: usize> {
    Dense(DenseSimd<SIZE>),
    Sparse(SparseSimd<SIZE>),
}

impl<const S: usize, const C: usize> DereferenceArithmetic<HybridSimd<S, C>> for &HybridSimd<S, C>{}


impl<const S: usize, const C: usize> SimdArr<S> for HybridSimd<S, C>
{
    fn new_from_array(arr: &[f32; S]) -> Self {
        if arr.iter().filter(|e| **e == 0.).count() > C {
            Self::Sparse(SparseSimd::new_from_array(arr))
        } else {
            Self::Dense(DenseSimd::new_from_array(arr))
        }
    }

    fn zero() -> Self {
        Self::Sparse(SparseSimd::zero())
    }

    fn to_array(&self) -> [f32; S] {
        match self {
            HybridSimd::Dense(d) => d.to_array(),
            HybridSimd::Sparse(s) => s.to_array(),
        }
    }

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self {
        HybridSimd::Sparse(SparseSimd::new_from_value_and_pos(val, pos))
    }
}

impl<const S: usize, const C: usize> Index<usize> for HybridSimd<S, C> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            HybridSimd::Dense(d) => &d[index],
            HybridSimd::Sparse(s) => &s[index],
        }
    }
}

impl<const S: usize, const C: usize> IndexMut<usize> for HybridSimd<S, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            HybridSimd::Dense(d) => &mut d[index],
            HybridSimd::Sparse(s) => &mut s[index],
        }
    }
}

impl<const S: usize, const C: usize> Add<&HybridSimd<S, C>> for HybridSimd<S, C>
{
    type Output = HybridSimd<S, C>;

    fn add(self, rhs: &HybridSimd<S, C>) -> Self::Output {
        &self + rhs
    }
}

impl<const S: usize, const C: usize> Sub<&HybridSimd<S, C>> for HybridSimd<S, C>
{
    type Output = HybridSimd<S, C>;

    fn sub(self, rhs: &HybridSimd<S, C>) -> Self::Output {
        &self - rhs
    }
}

impl<const S: usize, const C: usize> Add<&HybridSimd<S, C>> for &HybridSimd<S, C>
{
    type Output = HybridSimd<S, C>;

    fn add(self, rhs: &HybridSimd<S, C>) -> HybridSimd<S, C> {
        match (&self, rhs) {
            (HybridSimd::Dense(a), HybridSimd::Dense(b)) => HybridSimd::Dense(*a + b),
            (HybridSimd::Dense(a), HybridSimd::Sparse(ref b))
            | (HybridSimd::Sparse(ref b), HybridSimd::Dense(a)) => {
                HybridSimd::Dense(DenseSimd::new_from_array(&b.to_array()) + a)
            }
            (HybridSimd::Sparse(a), HybridSimd::Sparse(b)) => {
                let sparse_result = *a + b;
                if sparse_result.non_zero_count() > C {
                    HybridSimd::Sparse(sparse_result)
                } else {
                    HybridSimd::Dense(DenseSimd::new_from_array(&sparse_result.to_array()))
                }
            }
        }
    }
}

impl<const S: usize, const C: usize> Sub<&HybridSimd<S, C>> for &HybridSimd<S, C>
{
    type Output = HybridSimd<S, C>;

    fn sub(self, rhs: &HybridSimd<S, C>) -> Self::Output {
        match (self, rhs) {
            (HybridSimd::Dense(a), HybridSimd::Dense(b)) => HybridSimd::Dense(a - b),
            (HybridSimd::Dense(a), HybridSimd::Sparse(b)) => {
                HybridSimd::Dense(a - &DenseSimd::new_from_array(&b.to_array()))
            }
            (HybridSimd::Sparse(b), HybridSimd::Dense(a)) => {
                HybridSimd::Dense(DenseSimd::new_from_array(&b.to_array()) - a)
            }
            (HybridSimd::Sparse(a), HybridSimd::Sparse(b)) => {
                let sparse_result = *a - b;
                if sparse_result.non_zero_count() > C {
                    HybridSimd::Sparse(sparse_result)
                } else {
                    HybridSimd::Dense(DenseSimd::new_from_array(&sparse_result.to_array()))
                }
            }
        }
    }
}

impl<const S: usize, const C: usize> Mul<f32> for &HybridSimd<S, C> {
    type Output = HybridSimd<S, C>;

    fn mul(self, rhs: f32) -> Self::Output {
        match self {
            HybridSimd::Dense(d) => HybridSimd::Dense(d * rhs),
            HybridSimd::Sparse(s) => HybridSimd::Sparse(s * rhs),
        }
    }
}

impl<const S: usize, const C: usize> Div<f32> for &HybridSimd<S, C> {
    type Output = HybridSimd<S, C>;

    fn div(self, rhs: f32) -> Self::Output {
        match self {
            HybridSimd::Dense(d) => HybridSimd::Dense(d / rhs),
            HybridSimd::Sparse(s) => HybridSimd::Sparse(s / rhs),
        }
    }
}
