use crate::simd_arr::SimdArr;

use super::{check_nan, Dual};

pub trait ExtendedArithmetic {
    fn sqrt(self) -> Self;

    fn neg(self) -> Self;

    fn pow2(self) -> Self;

    fn abs(self) -> Self;

    fn sigmoid(self) -> Self;

    fn relu(self) -> Self;

    fn sqrt_on_mut(&mut self);
    fn neg_on_mut(&mut self);
    fn pow2_on_mut(&mut self);
    fn abs_on_mut(&mut self);
    fn sigmoid_on_mut(&mut self);
    fn relu_on_mut(&mut self);

    fn accumulate(&mut self, x: Self);
}

impl<const P: usize, S: SimdArr<P>> ExtendedArithmetic for Dual<P, S> {
    fn sqrt(mut self) -> Self {
        self.sqrt_on_mut();
        check_nan(self)
    }

    fn neg(mut self) -> Self {
        self.neg_on_mut();
        check_nan(self)
    }

    fn pow2(mut self) -> Self {
        self.pow2_on_mut();
        check_nan(self)
    }

    fn abs(mut self) -> Self {
        self.abs_on_mut();
        check_nan(self)
    }

    fn relu(mut self) -> Self {
        self.relu_on_mut();
        check_nan(self)
    }

    fn sigmoid(mut self) -> Self {
        self.sigmoid_on_mut();
        check_nan(self)
    }

    fn sqrt_on_mut(&mut self) {
        self.real = self.real.sqrt();
        self.sigma.multiply(1. / (2. * self.real.sqrt()));
    }

    fn neg_on_mut(&mut self) {
        self.real = -self.real;
        self.sigma.multiply(-1.);
    }

    fn pow2_on_mut(&mut self) {
        self.real *= self.real;
        self.sigma.multiply(self.real * 2.);
    }

    fn abs_on_mut(&mut self) {
        if self.real < 0. {
            self.real = -self.real;
            self.sigma.neg();
        }
    }

    fn sigmoid_on_mut(&mut self) {
        self.real = self.real.sigmoid();

        for i in 0..P {
            self.sigma[i] *= self.real * (1. - self.real);
        }
    }

    fn relu_on_mut(&mut self) {
        if self.real < 0. {
            self.real = 0.;
            self.sigma = S::zero();
        }
    }

    fn accumulate(&mut self, x: Dual<P, S>) {
        self.real += x.real;
        self.sigma.acumulate(&x.sigma);
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

    fn relu(self) -> Self {
        self.max(0.)
    }

    fn sigmoid(mut self) -> Self {
        self.sigmoid_on_mut();
        self
    }

    fn sqrt_on_mut(&mut self) {
        *self = self.sqrt()
    }

    fn neg_on_mut(&mut self) {
        *self = -*self;
    }

    fn pow2_on_mut(&mut self) {
        *self = *self * *self;
    }

    fn abs_on_mut(&mut self) {
        *self = self.abs();
    }

    fn sigmoid_on_mut(&mut self) {
        *self = 1. / (1. + (-*self).exp());
    }

    fn relu_on_mut(&mut self) {
        *self = self.max(0.);
    }

    fn accumulate(&mut self, x: f32) {
        *self += x;
    }
}
