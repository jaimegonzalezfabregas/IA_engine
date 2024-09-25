use crate::simd_arr::SimdArr;

use super::{check_nan, Dual};

pub trait ExtendedArithmetic {
    fn sqrt(self) -> Self;

    fn neg(self) -> Self;

    fn pow2(self) -> Self;

    fn abs(self) -> Self;
}

impl<const P: usize, S: SimdArr<P>> ExtendedArithmetic for Dual<P, S> {
    fn sqrt(mut self) -> Self {
        self.real = self.real.sqrt();
        self.sigma.multiply(1. / (2. * self.real.sqrt()));
        check_nan(self)
    }

    fn neg(mut self) -> Self {
        self.real = -self.real;
        self.sigma.multiply(-1.);
        check_nan(self)
    }

    fn pow2(mut self) -> Self {
        self.real *= self.real;
        self.sigma.multiply(self.real * 2.);
        check_nan(self)
    }

    fn abs(mut self) -> Self {
        if self.real < 0. {
            self.real = -self.real;
            self.sigma = self.sigma.neg();
        }

        check_nan(self)
    }
}

impl ExtendedArithmetic for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn neg(self) -> Self {
        -self
    }

    fn pow2(self) -> Self {
        self * self
    }

    fn abs(self) -> Self {
        self.abs()
    }
}
