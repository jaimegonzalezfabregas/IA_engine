use std::ops::Add;

use crate::simd_arr::{DereferenceArithmetic, SimdArr};

use super::{assert_finite, Dual};

impl<const P: usize, S: SimdArr<P>> Add<&Dual<P, S>> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn add(self, rhs: &Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs.real,
            sigma: self.sigma + &rhs.sigma,
        })
    }
}


impl<const P: usize, S: SimdArr<P>> Add<f32> for Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn add(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs,
            sigma: self.sigma,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Add<&Dual<P, S>> for &Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn add(self, rhs: &Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs.real,
            sigma: &self.sigma + &rhs.sigma,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Add<f32> for &Dual<P, S>
where
    for<'own> &'own S: DereferenceArithmetic<S>,
{
    type Output = Dual<P, S>;

    fn add(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs,
            sigma: self.sigma.clone()
        })
    }
}
