use std::ops::Add;

use crate::simd_arr::SimdArr;

use super::Dual;

impl<const P: usize, S: SimdArr<P>> Add<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn add(mut self, rhs: Dual<P, S>) -> Self::Output {
        self.real += rhs.real;
        self.sigma.acumulate(&rhs.sigma);

        self
    }
}

impl<const P: usize, S: SimdArr<P>> Add<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn add(mut self, rhs: f32) -> Self::Output {
        self.real += rhs;

        self
    }
}