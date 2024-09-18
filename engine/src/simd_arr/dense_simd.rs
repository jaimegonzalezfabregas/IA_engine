use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use super::{DereferenceArithmetic, SimdArr};

#[derive(Clone, Copy, Debug)]
pub struct DenseSimd<const S: usize> {
    data: [f32; S],
}

impl<const S: usize> DereferenceArithmetic<DenseSimd<S>> for &DenseSimd<S> {}

impl<const S: usize> SimdArr<S> for DenseSimd<S>
{
    fn new_from_array(data: &[f32; S]) -> DenseSimd<S> {
        Self { data: *data }
    }

    fn zero() -> DenseSimd<S> {
        Self { data: [0.; S] }
    }

    fn to_array(&self) -> [f32; S] {
        self.data
    }

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self {
        let mut ret = Self { data: [0.; S] };
        ret.data[pos] = val;
        ret
    }
}

impl<const S: usize> Index<usize> for DenseSimd<S> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const S: usize> IndexMut<usize> for DenseSimd<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const S: usize> Add<&DenseSimd<S>> for DenseSimd<S> {
    type Output = DenseSimd<S>;

    fn add(self, rhs: &DenseSimd<S>) -> Self::Output {
        &self + rhs
    }
}

impl<const S: usize> Sub<&DenseSimd<S>> for DenseSimd<S> {
    type Output = DenseSimd<S>;

    fn sub(self, rhs: &DenseSimd<S>) -> Self::Output {
        &self - rhs
    }
}

impl<const S: usize> Add<&DenseSimd<S>> for &DenseSimd<S> {
    type Output = DenseSimd<S>;

    fn add(self, rhs: &DenseSimd<S>) -> Self::Output {
        let mut ret = DenseSimd { data: [0.; S] };
        for (i, (a, b)) in self.data.into_iter().zip(rhs.data.into_iter()).enumerate() {
            ret.data[i] = a + b;
        }
        ret
    }
}

impl<const S: usize> Sub<&DenseSimd<S>> for &DenseSimd<S> {
    type Output = DenseSimd<S>;

    fn sub(self, rhs: &DenseSimd<S>) -> Self::Output {
        let mut ret = DenseSimd { data: [0.; S] };
        for (i, (a, b)) in self.data.into_iter().zip(rhs.data.into_iter()).enumerate() {
            ret.data[i] = a - b;
        }
        ret
    }
}

impl<const S: usize> Mul<f32> for &DenseSimd<S> {
    type Output = DenseSimd<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut ret = DenseSimd { data: [0.; S] };

        for (i, a) in self.data.into_iter().enumerate() {
            ret.data[i] = a * rhs;
        }
        ret
    }
}

impl<const S: usize> Div<f32> for &DenseSimd<S> {
    type Output = DenseSimd<S>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut ret = DenseSimd { data: [0.; S] };

        for (i, a) in self.data.into_iter().enumerate() {
            ret.data[i] = a / rhs;
        }
        ret
    }
}
