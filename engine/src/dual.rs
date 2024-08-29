use core::ops::*;

#[derive(Clone, Copy, Debug)]

pub struct Dual<const P: usize> {
    real: f32,
    sigma: [f32; P],
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

impl<const P: usize> Default for Dual<P> {
    fn default() -> Self {
        Self::cero()
    }
}

impl<const P: usize> Add<&Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn add(self, rhs: &Dual<P>) -> Self::Output {
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

impl<const P: usize> AddAssign<&Dual<P>> for Dual<P> {
    fn add_assign(&mut self, rhs: &Dual<P>) {
        *self = *self + rhs;
    }
}

impl<const P: usize> Sub<&Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn sub(self, rhs: &Dual<P>) -> Self::Output {
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

impl<const P: usize> SubAssign<&Dual<P>> for Dual<P> {
    fn sub_assign(&mut self, rhs: &Dual<P>) {
        *self = *self - rhs;
    }
}

impl<const P: usize> Mul<&Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn mul(self, rhs: &Dual<P>) -> Self::Output {
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

impl<const P: usize> MulAssign<&Dual<P>> for Dual<P> {
    fn mul_assign(&mut self, rhs: &Dual<P>) {
        *self = *self * rhs;
    }
}

impl<const P: usize> Div<&Dual<P>> for Dual<P> {
    type Output = Dual<P>;

    fn div(self, rhs: &Dual<P>) -> Self::Output {
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

impl<const P: usize> DivAssign<&Dual<P>> for Dual<P> {
    fn div_assign(&mut self, rhs: &Dual<P>) {
        *self = *self / rhs;
    }
}

impl<const P: usize> Neg for Dual<P> {
    type Output = Dual<P>;

    fn neg(self) -> Self::Output {
        self * &Dual {
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
