use std::ops::{Index, IndexMut};

use super::{dense_simd::DenseSimd, sparse_simd::SparseSimd, SimdArr};

#[derive(Clone, Copy, Debug)]
pub enum HybridSimd<const SIZE: usize, const CRITIALITY: usize> {
    Dense(DenseSimd<SIZE>),
    Sparse(SparseSimd<CRITIALITY, SIZE>),
}

impl<const S: usize, const C: usize> SimdArr<S> for HybridSimd<S, C> {
    fn new_from_array(arr: [f32; S]) -> Self {
        match SparseSimd::new_from_array(&arr) {
            None => HybridSimd::Dense(DenseSimd::new_from_array(arr)),
            Some(sparse) => HybridSimd::Sparse(sparse),
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

    fn neg(self) -> Self {
        match self {
            HybridSimd::Dense(d) => HybridSimd::Dense(d.neg()),
            HybridSimd::Sparse(s) => HybridSimd::Sparse(s.neg()),
        }
    }

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self {
        HybridSimd::Sparse(SparseSimd::new_from_value_and_pos(val, pos))
    }

    fn acumulate(&mut self, rhs: &Self) {
        match (&self, rhs) {
            (HybridSimd::Dense(mut a), HybridSimd::Dense(b)) => a.acumulate(b),
            (HybridSimd::Dense(mut a), HybridSimd::Sparse(b)) => {
                let transformation = DenseSimd::new_from_array(b.to_array());
                a.acumulate(&transformation);
            }
            (HybridSimd::Sparse(a), HybridSimd::Dense(b)) => {
                let mut transformation = DenseSimd::new_from_array(a.to_array());
                transformation.acumulate(b);

                *self = HybridSimd::Dense(transformation);
            }
            (HybridSimd::Sparse(mut a), HybridSimd::Sparse(b)) => {
                a.acumulate(b);
            }
        }
    }

    fn multiply(&mut self, rhs: f32) {
        match self {
            HybridSimd::Dense(d) => d.multiply(rhs),
            HybridSimd::Sparse(s) => s.multiply(rhs),
        }
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
