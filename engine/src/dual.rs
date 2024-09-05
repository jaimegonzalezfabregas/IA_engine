use core::ops::*;

use number_traits::{One, Sqrt, Zero};

use crate::{dense_simd::DenseSimd, hybrid_simd::HybridSimd, sparse_simd::SparseSimd};

type Simd<const P: usize> = SparseSimd<P>;

#[derive(Clone, Copy, Debug)]

pub struct Dual<const P: usize> {
    real: f32,
    sigma: Simd<P>,
}

impl<const P: usize> From<f32> for Dual<P> {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl<const P: usize> Dual<P> {
    pub fn cero() -> Self {
        Self {
            real: 0.,
            sigma: Simd::zero(),
        }
    }

    pub fn get_gradient(&self) -> [f32; P] {
        self.sigma.to_array()
    }

    pub fn get_real(&self) -> f32 {
        self.real
    }

    pub fn new_param(real: f32, i: usize) -> Dual<P> {
        let mut ret = Self::cero();
        ret.real = real;
        ret.sigma[i] = 1.;
        ret
    }

    pub fn new(real: f32) -> Self {
        let mut ret = Self::cero();
        ret.real = real;
        ret
    }

    pub fn new_full(real: f32, sigma: &[f32; P]) -> Self {
        Self {
            real,
            sigma: Simd::new_from_array(sigma),
        }
    }

    pub fn abs(&self) -> Dual<P> {
        if *self < 0. {
            -*self
        } else {
            *self
        }
    }
}

impl<const P: usize> Add<Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn add(self, rhs: Dual<P>) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs.real,
            sigma: self.sigma + rhs.sigma,
        })
    }
}

impl<const P: usize> Add<f32> for Dual<P> {
    type Output = Dual<P>;

    fn add(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real + rhs,
            sigma: self.sigma,
        })
    }
}

impl<const P: usize> AddAssign<Dual<P>> for Dual<P> {
    fn add_assign(&mut self, rhs: Dual<P>) {
        *self = *self + rhs;
    }
}

impl<const P: usize> Sub<Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn sub(self, rhs: Dual<P>) -> Self::Output {
        assert_finite(Dual {
            real: self.real - rhs.real,
            sigma: self.sigma - rhs.sigma,
        })
    }
}

impl<const P: usize> Sub<f32> for Dual<P> {
    type Output = Dual<P>;

    fn sub(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real - rhs,
            sigma: self.sigma,
        })
    }
}

impl<const P: usize> SubAssign<Dual<P>> for Dual<P> {
    fn sub_assign(&mut self, rhs: Dual<P>) {
        *self = *self - rhs;
    }
}

impl<const P: usize> Mul<Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn mul(self, rhs: Dual<P>) -> Self::Output {
        assert_finite(Dual {
            real: self.real * rhs.real,
            sigma: (rhs.sigma * self.real) + (self.sigma * rhs.real),
        })
    }
}

impl<const P: usize> Mul<f32> for Dual<P> {
    type Output = Dual<P>;

    fn mul(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real * rhs,
            sigma: self.sigma * rhs,
        })
    }
}

impl<const P: usize> MulAssign<Dual<P>> for Dual<P> {
    fn mul_assign(&mut self, rhs: Dual<P>) {
        *self = *self * rhs;
    }
}

impl<const P: usize> Div<Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn div(self, rhs: Dual<P>) -> Self::Output {
        assert_finite(Dual {
            real: self.real / rhs.real,
            sigma: ((self.sigma * rhs.real) - (rhs.sigma * self.real)) / (rhs.real * rhs.real),
        })
    }
}

impl<const P: usize> Div<f32> for Dual<P> {
    type Output = Dual<P>;

    fn div(self, rhs: f32) -> Self::Output {
        assert_finite(Dual {
            real: self.real / rhs,
            sigma: self.sigma / rhs,
        })
    }
}

impl<const P: usize> DivAssign<Dual<P>> for Dual<P> {
    fn div_assign(&mut self, rhs: Dual<P>) {
        *self = *self / rhs;
    }
}

impl<const P: usize> Neg for Dual<P> {
    type Output = Dual<P>;

    fn neg(self) -> Self::Output {
        self * Dual::from(-1.)
    }
}

impl<const P: usize> PartialEq for Dual<P> {
    fn eq(&self, other: &Self) -> bool {
        self.real.eq(&other.real)
    }
}

impl<const P: usize> PartialOrd for Dual<P> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

// extended traits for https://docs.rs/number_traits/latest/number_traits/index.html

impl<const P: usize> Sqrt for Dual<P> {
    fn sqrt(&self) -> Self {
        assert_finite(Dual {
            real: self.real.sqrt(),
            sigma: self.sigma / (2. * self.real.sqrt()),
        })
    }
}

impl<const P: usize> Zero for Dual<P> {
    fn zero() -> Self {
        Self {
            real: 0.,
            sigma: Simd::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero()
    }
}

impl<const P: usize> One for Dual<P> {
    fn one() -> Self {
        Self {
            real: 1.,
            sigma: Simd::zero(),
        }
    }

    fn is_one(&self) -> bool {
        self.real.is_one()
    }
}

impl<const P: usize> Eq for Dual<P> {}

impl<const P: usize> PartialEq<f32> for Dual<P> {
    fn eq(&self, other: &f32) -> bool {
        self.real.eq(other)
    }
}

impl<const P: usize> PartialOrd<f32> for Dual<P> {
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(other)
    }
}

impl<const P: usize> From<Dual<P>> for f32 {
    fn from(value: Dual<P>) -> Self {
        value.get_real()
    }
}

fn assert_finite<const P: usize>(a: Dual<P>) -> Dual<P> {
    assert!(a.get_real().is_finite());
    assert!(a
        .get_gradient()
        .iter()
        .fold(true, |acc, x| acc && x.is_finite()));
    a
}
