use std::ops::Div;

use crate::simd_arr::{DereferenceArithmetic, SimdArr};

use super::{assert_finite, Dual};

impl<const P: usize, S: SimdArr<P>> Div<&Dual<P, S>> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn div(self, rhs: &Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real / rhs.real,
            sigma: &((&self.sigma * rhs.real) - &(&rhs.sigma * self.real))
                / (rhs.real * rhs.real),
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Div<f32> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn div(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real / rhs,
            sigma: &self.sigma / rhs,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Div<&Dual<P, S>> for &Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn div(self, rhs: &Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real / rhs.real,
            sigma: &((&self.sigma * rhs.real) - &(&rhs.sigma * self.real))
                / (rhs.real * rhs.real),
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Div<f32> for &Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn div(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real / rhs,
            sigma: &self.sigma / rhs,
        })
    }
}

