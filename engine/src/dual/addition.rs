use std::ops::{Add, AddAssign};

use crate::simd_arr::SimdArr;

use super::{assert_finite, Dual};


impl<const P: usize, S: SimdArr<P>> Add<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn add(self, rhs: Dual<P, S>) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs.real,
            sigma: self.sigma + &rhs.sigma,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> Add<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn add(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs,
            sigma: self.sigma,
        })
    }
}

impl<const P: usize, S: SimdArr<P>> AddAssign<Dual<P, S>> for Dual<P, S> {
    fn add_assign(&mut self, rhs: Dual<P, S>) {
        *self =  assert_finite(Dual {
            real: self.real + rhs.real,
            sigma: self.sigma + &rhs.sigma,
        })
    }
}