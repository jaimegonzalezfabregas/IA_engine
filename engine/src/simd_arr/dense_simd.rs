use std::ops::{Index, IndexMut};

use super::SimdArr;

#[derive(Clone, Debug)]
pub struct DenseSimd<const S: usize> {
    data: [f32; S],
}

impl<const S: usize> SimdArr<S> for DenseSimd<S> {
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

    fn neg(mut self) -> Self {
        for i in 0..S {
            self.data[i] *= -1.;
        }
        self
    }

    fn acumulate(&mut self, rhs: &Self) {
        for i in 0..S {
            self.data[i] += rhs[i];
        }
    }

    fn multiply(&mut self, rhs: f32) {
        for x in &mut self.data {
            *x *= rhs;
        }
    }

    fn new_from_array(data: [f32; S]) -> DenseSimd<S> {
        Self { data }
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
