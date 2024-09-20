mod addition;
mod division;
mod multiplication;
mod substraction;

use core::ops::*;

use number_traits::{One, Sqrt, Zero};

use crate::simd_arr::{DereferenceArithmetic, SimdArr};

#[derive(Clone, Debug)]

pub struct Dual<const P: usize, S: SimdArr<P>>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    real: f32,
    sigma: S,
}

impl<const P: usize, S: SimdArr<P>> From<f32> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl<const P: usize, S: SimdArr<P>> Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    pub fn new_param(real: f32, i: usize) -> Dual<P, S> {
        Self {
            real: real,
            sigma: S::new_from_value_and_pos(1., i),
        }
    }

    pub(crate) fn set_real(&mut self, val: f32) {
        self.real = val
    }
}

impl<const P: usize, S: SimdArr<P>> Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
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

impl<const P: usize, S: SimdArr<P>> Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    pub fn abs(&mut self) {
        if self.real < 0. {
            self.real = -self.real;
            self.sigma.neg();
        }
    }
}

impl<const P: usize, S: SimdArr<P>> Neg for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}

impl<const P: usize, S: SimdArr<P>> PartialEq for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn eq(&self, other: &Self) -> bool {
        self.real.eq(&other.real)
    }
}

impl<const P: usize, S: SimdArr<P>> PartialOrd for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

// extended traits for https://docs.rs/number_traits/latest/number_traits/index.html

impl<const P: usize, S: SimdArr<P>> Sqrt for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn sqrt(&self) -> Self {
        assert_finite(Dual {
            real: self.real.sqrt(),
            sigma: &self.sigma / (2. * self.real.sqrt()),
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Zero for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn zero() -> Self {
        Self {
            real: 0.,
            sigma: SimdArr::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero()
    }
}

impl<const P: usize, S: SimdArr<P>> One for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn one() -> Self {
        Self {
            real: 1.,
            sigma: SimdArr::zero(),
        }
    }

    fn is_one(&self) -> bool {
        self.real.is_one()
    }
}

impl<const P: usize, S: SimdArr<P>> Eq for Dual<P, S> where
    for<'own> &'own S: DereferenceArithmetic<S>
{
}

impl<const P: usize, S: SimdArr<P>> PartialEq<f32> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn eq(&self, other: &f32) -> bool {
        self.real.eq(other)
    }
}

impl<const P: usize, S: SimdArr<P>> PartialOrd<f32> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(other)
    }
}

impl<const P: usize, S: SimdArr<P>> From<Dual<P, S>> for f32
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn from(value: Dual<P, S>) -> Self {
        value.get_real()
    }
}

const fn assert_finite<const P: usize, S: SimdArr<P>>(a: Dual<P, S>) -> Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    // assert!(a.get_real().is_finite());
    // assert!(a
    //     .get_gradient()
    //     .iter()
    //     .fold(true, |acc, x| acc && x.is_finite()));
    a
}
