use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use crate::{dense_simd::DenseSimd, sparse_simd::SparseSimd};

#[derive(Clone, Copy, Debug)]
pub enum HybridSimd<const S: usize> {
    Dense(DenseSimd<S>),
    Sparse(SparseSimd<S>),
}

impl<const S: usize> HybridSimd<S> {
    pub(crate) const fn zero() -> Self {
        Self::Sparse(SparseSimd::zero())
    }

    pub(crate) fn to_array(&self) -> [f32; S] {
        match self {
            HybridSimd::Dense(d) => d.to_array(),
            HybridSimd::Sparse(s) => s.to_array(),
        }
    }

    pub(crate) fn new_from_array(arr: &[f32; S]) -> Self {
        Self::Sparse(SparseSimd::new_from_array(arr)).spontaneous_decay()
    }

    fn spontaneous_decay(self) -> Self {
        match self {
            HybridSimd::Dense(_) => self,
            HybridSimd::Sparse(x) => {
                if x.non_zero_count() > S * 2 / 3 {
                    self.decay()
                } else {
                    self
                }
            }
        }
    }

    fn decay(self) -> Self {
        HybridSimd::Dense(DenseSimd::new_from_array(&self.to_array()))
    }
}

impl<const S: usize> Index<usize> for HybridSimd<S> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            HybridSimd::Dense(d) => &d[index],
            HybridSimd::Sparse(s) => &s[index],
        }
    }
}

impl<const S: usize> IndexMut<usize> for HybridSimd<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            HybridSimd::Dense(d) => &mut d[index],
            HybridSimd::Sparse(s) => &mut s[index],
        }
    }
}

impl<const S: usize> Add<HybridSimd<S>> for HybridSimd<S> {
    type Output = HybridSimd<S>;

    fn add(self, rhs: HybridSimd<S>) -> Self::Output {
        match (self, rhs) {
            (HybridSimd::Dense(a), HybridSimd::Dense(b)) => Self::Dense(a + b),
            (b @ HybridSimd::Dense(_), a @ HybridSimd::Sparse(_)) => a.decay() + b,
            (b @ HybridSimd::Sparse(_), a @ HybridSimd::Dense(_)) => a + b.decay(),
            (HybridSimd::Sparse(a), HybridSimd::Sparse(b)) => Self::Sparse(a + b),
        }.spontaneous_decay()
    }
}

impl<const S: usize> Sub<HybridSimd<S>> for HybridSimd<S> {
    type Output = HybridSimd<S>;

    fn sub(self, rhs: HybridSimd<S>) -> Self::Output {
        match (self, rhs) {
            (HybridSimd::Dense(a), HybridSimd::Dense(b)) => Self::Dense(a - b),
            (a @ HybridSimd::Dense(_), b @ HybridSimd::Sparse(_)) => a - b.decay(),
            (a @ HybridSimd::Sparse(_), b @ HybridSimd::Dense(_)) => a.decay() - b,
            (HybridSimd::Sparse(a), HybridSimd::Sparse(b)) => Self::Sparse(a - b),
        }.spontaneous_decay()
    }
}

impl<const S: usize> Mul<HybridSimd<S>> for HybridSimd<S> {
    type Output = HybridSimd<S>;

    fn mul(self, rhs: HybridSimd<S>) -> Self::Output {
        match (self, rhs) {
            (HybridSimd::Dense(a), HybridSimd::Dense(b)) => Self::Dense(a * b),
            (a @ HybridSimd::Dense(_), b @ HybridSimd::Sparse(_)) => a * b.decay(),
            (a @ HybridSimd::Sparse(_), b @ HybridSimd::Dense(_)) => a.decay() * b,
            (HybridSimd::Sparse(a), HybridSimd::Sparse(b)) => Self::Sparse(a * b),
        }.spontaneous_decay()
    }
}

impl<const S: usize> Mul<f32> for HybridSimd<S> {
    type Output = HybridSimd<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        match self {
            HybridSimd::Dense(d) => HybridSimd::Dense(d * rhs),
            HybridSimd::Sparse(s) => HybridSimd::Sparse(s * rhs),
        }
    }
}


impl<const S: usize> Div<f32> for HybridSimd<S> {
    type Output = HybridSimd<S>;

    fn div(self, rhs: f32) -> Self::Output {
        match self {
            HybridSimd::Dense(d) => HybridSimd::Dense(d / rhs),
            HybridSimd::Sparse(s) => HybridSimd::Sparse(s / rhs),
        }
    }
}
