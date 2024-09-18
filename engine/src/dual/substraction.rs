use std::ops::{Add, Div, Mul, Sub, SubAssign};

use crate::simd_arr::{DereferenceArithmetic, SimdArr};

use super::{assert_finite, Dual};

impl<const P: usize, S: SimdArr<P>> Sub<&Dual<P, S>> for &Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn sub(self, rhs: &Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real - rhs.real,
            sigma: &self.sigma - &rhs.sigma,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Sub<f32> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn sub(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real - rhs,
            sigma: self.sigma,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> SubAssign<Dual<P, S>> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    fn sub_assign(&mut self, rhs: Dual<P, S>) {
        *self = &*self - &rhs;
    }
}
