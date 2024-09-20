use std::ops::Mul;

use crate::simd_arr::{DereferenceArithmetic, SimdArr};

use super::{assert_finite, Dual};

impl<const P: usize, S: SimdArr<P>> Mul<&Dual<P, S>> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn mul(self, rhs: &Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real * rhs.real,
            sigma: (&rhs.sigma * self.real) + &(&self.sigma * rhs.real),
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Mul<f32> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn mul(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real * rhs,
            sigma: &self.sigma * rhs,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Mul<&Dual<P, S>> for &Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn mul(self, rhs: &Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real * rhs.real,
            sigma: (&rhs.sigma * self.real) + &(&self.sigma * rhs.real),
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Mul<f32> for &Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn mul(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real * rhs,
            sigma: &self.sigma * rhs,
        })
    }
}

