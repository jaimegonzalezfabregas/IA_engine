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

    pub fn new_full(real: f32, sigma: [f32; P]) -> Self {
        Self {
            real,
            sigma: SimdArr::new_from_array(sigma),
        }
    }
}

impl<const P: usize, S: SimdArr<P>> Dual<P, S> {
    pub fn abs(mut self) -> Self {
        if self.real < 0. {
            self.real = -self.real;
            self.sigma = self.sigma.neg();
        }
        self
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

impl<const P: usize, S: SimdArr<P>> Sub<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn sub(mut self, rhs: Dual<P, S>) -> Self::Output {
        self.real -= &rhs.real;
        self.sigma.acumulate(&rhs.sigma.neg());

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

impl<const P: usize, S: SimdArr<P>> Mul<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn mul(mut self, mut rhs: Dual<P, S>) -> Self::Output {
        self.real *= rhs.real;

        self.sigma.multiply(rhs.real);
        rhs.sigma.multiply(self.real);

        self.sigma.acumulate(&rhs.sigma);

        self
    }
}

impl<const P: usize, S: SimdArr<P>> Mul<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.real /= rhs;

        self.sigma.multiply(1. / rhs);

        self
    }
}

impl<const P: usize, S: SimdArr<P>> Div<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn div(mut self, mut rhs: Dual<P, S>) -> Self::Output {
        self.real /= rhs.real;

        self.sigma.multiply(rhs.real);
        rhs.sigma.multiply(self.real);

        self.sigma.acumulate(&rhs.sigma.neg());

        self.sigma.multiply(1. / (rhs.real * rhs.real));

        self
    }
}

impl<const P: usize, S: SimdArr<P>> Div<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn div(mut self, rhs: f32) -> Self::Output {
        self.real /= rhs;

        self.sigma.multiply(1. / rhs);

        self
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

pub trait ExtendedArithmetic {
    fn sqrt(self) -> Self;

    fn neg(self) -> Self;

    fn pow2(self) -> Self;
}

impl<const P: usize, S: SimdArr<P>> ExtendedArithmetic for Dual<P, S> {
    fn sqrt(mut self) -> Self {
        self.real = self.real.sqrt();
        self.sigma.multiply(1. / (2. * self.real.sqrt()));
        self
    }

    fn neg(mut self) -> Self {
        self.real = -self.real;
        self.sigma.multiply(-1.);
        self
    }

    fn pow2(mut self) -> Self {
        self.real *= self.real;
        self.sigma.multiply(self.real * 2.);
        self
    }
}
