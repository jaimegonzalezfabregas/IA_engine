use core::ops::*;

use number_traits::{One, Pow, Sqrt, Zero};

#[derive(Clone, Copy, Debug)]

pub struct Dual<const P: usize> {
    real: f32,
    sigma: [f32; P],
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
            sigma: [0.; P],
        }
    }

    pub fn get_gradient(&self) -> [f32; P] {
        self.sigma
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
}

impl<const P: usize> Add<Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn add(self, rhs: Dual<P>) -> Self::Output {
        let mut ret = Dual {
            real: self.real + rhs.real,
            sigma: [0.; P],
        };
        for i in 0..P {
            ret.sigma[i] = self.sigma[i] + rhs.sigma[i]
        }

        ret
    }
}

impl<const P: usize, N: Into<f32>> Add<N> for Dual<P> {
    type Output = Dual<P>;

    fn add(self, rhs: N) -> Self::Output {
        Dual {
            real: self.real + rhs.into(),
            sigma: self.sigma,
        }
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
        let mut ret = Dual {
            real: self.real - rhs.real,
            sigma: [0.; P],
        };
        for i in 0..P {
            ret.sigma[i] = self.sigma[i] - rhs.sigma[i]
        }

        ret
    }
}

impl<const P: usize, N: Into<f32>> Sub<N> for Dual<P> {
    type Output = Dual<P>;

    fn sub(self, rhs: N) -> Self::Output {
        Dual {
            real: self.real - rhs.into(),
            sigma: self.sigma,
        }
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
        let mut ret = Dual {
            real: self.real * rhs.real,
            sigma: [0.; P],
        };
        for i in 0..P {
            ret.sigma[i] = self.real * rhs.sigma[i] + self.sigma[i] * rhs.real
        }

        ret
    }
}

impl<const P: usize, N: Into<f32> + Copy> Mul<N> for Dual<P> {
    type Output = Dual<P>;

    fn mul(self, rhs: N) -> Self::Output {
        let mut ret = Dual {
            real: self.real * rhs.into(),
            sigma: [0.; P],
        };

        for i in 0..P {
            ret.sigma[i] = self.sigma[i] * rhs.into()
        }

        ret
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
        let rhs_real_to_2 = rhs.real * rhs.real;

        let mut ret = Dual {
            real: self.real / rhs.real,
            sigma: [0.; P],
        };
        for i in 0..P {
            ret.sigma[i] = (self.sigma[i] * rhs.real - self.real * rhs.sigma[i]) / rhs_real_to_2
        }

        ret
    }
}

impl<const P: usize, N: Into<f32>> Div<N> for Dual<P> {
    type Output = Dual<P>;

    fn div(self, rhs: N) -> Self::Output {
        let rhs_real = rhs.into();
        let rhs_real_to_2 = rhs_real * rhs_real;

        let mut ret = Dual {
            real: self.real / rhs_real,
            sigma: [0.; P],
        };
        for i in 0..P {
            ret.sigma[i] = (self.sigma[i] * rhs_real) / rhs_real_to_2
        }

        ret
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
        self * Dual {
            real: -1.,
            sigma: [0.; P],
        }
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
        let real_sqrt = self.real.sqrt();

        let mut ret = Dual {
            real: real_sqrt,
            sigma: [0.; P],
        };

        let double_real_sqrt = 2. * real_sqrt;

        for i in 0..P {
            ret.sigma[i] = self.sigma[i] / double_real_sqrt;
        }

        ret
    }
}

impl<const P: usize> Zero for Dual<P> {
    fn zero() -> Self {
        Self {
            real: 0.,
            sigma: [0.; P],
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
            sigma: [0.; P],
        }
    }

    fn is_one(&self) -> bool {
        self.real.is_one()
    }
}
