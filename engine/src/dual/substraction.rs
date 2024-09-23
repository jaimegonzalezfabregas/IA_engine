use std::ops::Sub;

use crate::simd_arr::SimdArr;

use super::Dual;

impl<const P: usize, S: SimdArr<P>> Sub<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn sub(mut self, rhs: Dual<P, S>) -> Self::Output {
        println!("sub start: {self:?} - {rhs:?}");
        self.real -= &rhs.real;
        self.sigma.acumulate(&rhs.sigma.neg());
        println!("sub end sigma: {:?}", self.sigma);

        self
    }
}

impl<const P: usize, S: SimdArr<P>> Sub<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn sub(mut self, rhs: f32) -> Self::Output {
        self.real -= rhs;

        self
    }
}
