use core::ops::*;

use crate::simd_arr::SimdArr;

#[derive(Clone, Debug)]

pub struct Dual<const P: usize, S: SimdArr<P>> {
    real: f32,
    sigma: S,
}

impl<const P: usize, S: SimdArr<P>> From<f32> for Dual<P, S> {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl<const P: usize, S: SimdArr<P>> Dual<P, S> {
    pub fn new_param(real: f32, i: usize) -> Dual<P, S> {
        Self {
            real: real,
            sigma: S::new_from_value_and_pos(1., i),
        }
    }

    pub(crate) fn set_real(&mut self, val: f32) {
        self.real = val
    }

    fn sqrt(&mut self) {
        self.real = self.real.sqrt();
        self.sigma.multiply(1. / (2. * self.real.sqrt()))
    }

    fn acumulate(&mut self, rhs: Dual<P, S>) {
        self.real += rhs.real;
        self.sigma.acumulate(&rhs.sigma);
    }

    fn div(&mut self, mut rhs: Dual<P, S>) {
        self.real /= rhs.real;

        self.sigma.multiply(rhs.real);
        rhs.sigma.multiply(self.real);

        rhs.sigma.neg();

        self.sigma.acumulate(&mut rhs.sigma);

        self.sigma.multiply(1. / (rhs.real * rhs.real));
    }

    fn mul(&mut self, mut rhs: Dual<P, S>) {
        self.real *= rhs.real;

        self.sigma.multiply(rhs.real);
        rhs.sigma.multiply(self.real);

        self.sigma.acumulate(&rhs.sigma);
    }

    fn sub(&mut self, mut rhs: Dual<P, S>) {
        self.real -= rhs.real;
        rhs.sigma.neg();

        self.sigma.acumulate(&rhs.sigma);
    }

    fn neg(&mut self) {
        self.real = -self.real;
        self.sigma.multiply(-1.);
    }
}

impl<const P: usize, S: SimdArr<P>> Dual<P, S> {
    pub fn zero() -> Self {
        Self {
            real: 0.,
            sigma: S::zero(),
        }
    }

    pub fn get_gradient(&self) -> [f32; P] {
        self.sigma.to_array()
    }

    pub fn get_real(&self) -> f32 {
        self.real
    }

    pub fn new(real: f32) -> Self {
        let mut ret = Self::zero();
        ret.real = real;
        ret
    }

    pub fn new_full(real: f32, sigma: &[f32; P]) -> Self {
        Self {
            real,
            sigma: SimdArr::new_from_array(sigma),
        }
    }
}

impl<const P: usize, S: SimdArr<P>> Dual<P, S> {
    pub fn abs(&mut self) {
        if self.real < 0. {
            self.real = -self.real;
            self.sigma.neg();
        }
    }
}

impl<const P: usize, S: SimdArr<P>> PartialEq for Dual<P, S> {
    fn eq(&self, other: &Self) -> bool {
        self.real.eq(&other.real)
    }
}

impl<const P: usize, S: SimdArr<P>> PartialOrd for Dual<P, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

// extended traits for https://docs.rs/number_traits/latest/number_traits/index.html

impl<const P: usize, S: SimdArr<P>> Eq for Dual<P, S> {}

impl<const P: usize, S: SimdArr<P>> PartialEq<f32> for Dual<P, S> {
    fn eq(&self, other: &f32) -> bool {
        self.real.eq(other)
    }
}

impl<const P: usize, S: SimdArr<P>> PartialOrd<f32> for Dual<P, S> {
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(other)
    }
}

impl<const P: usize, S: SimdArr<P>> From<Dual<P, S>> for f32 {
    fn from(value: Dual<P, S>) -> Self {
        value.get_real()
    }
}
