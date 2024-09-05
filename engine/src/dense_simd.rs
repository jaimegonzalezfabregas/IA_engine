use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseSimd<const S: usize> {
    data: [f32; S],
}
impl<const S: usize> DenseSimd<S> {
    pub(crate) const fn new_from_array(data: &[f32; S]) -> DenseSimd<S> {
        Self { data: *data }
    }

    pub(crate) const fn zero() -> DenseSimd<S> {
        Self { data: [0.; S] }
    }

    pub(crate) const fn to_array(&self) -> [f32; S] {
        self.data
    }
}

impl<const S: usize> Index<usize> for DenseSimd<S> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const S: usize> IndexMut<usize> for DenseSimd<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const S: usize> Add for DenseSimd<S> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, (a, b)) in self.data.into_iter().zip(rhs.data.into_iter()).enumerate() {
            ret.data[i] = a + b;
        }
        ret
    }
}

impl<const S: usize> Add<f32> for DenseSimd<S> {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, a) in self.data.into_iter().enumerate() {
            ret.data[i] = a + rhs;
        }
        ret
    }
}

impl<const S: usize> Add<DenseSimd<S>> for f32 {
    type Output = DenseSimd<S>;

    fn add(self, rhs: DenseSimd<S>) -> Self::Output {
        rhs + self
    }
}

impl<const S: usize> Sub for DenseSimd<S> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, (a, b)) in self.data.into_iter().zip(rhs.data.into_iter()).enumerate() {
            ret.data[i] = a - b;
        }
        ret
    }
}

impl<const S: usize> Sub<f32> for DenseSimd<S> {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, a) in self.data.into_iter().enumerate() {
            ret.data[i] = a - rhs;
        }
        ret
    }
}

impl<const S: usize> Sub<DenseSimd<S>> for f32 {
    type Output = DenseSimd<S>;

    fn sub(self, rhs: DenseSimd<S>) -> Self::Output {
        let mut ret = DenseSimd { data: [0.; S] };
        for (i, a) in rhs.data.into_iter().enumerate() {
            ret.data[i] = self - a;
        }
        ret
    }
}

impl<const S: usize> Mul for DenseSimd<S> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, (a, b)) in self.data.into_iter().zip(rhs.data.into_iter()).enumerate() {
            ret.data[i] = a * b;
        }
        ret
    }
}

impl<const S: usize> Mul<f32> for DenseSimd<S> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, a) in self.data.into_iter().enumerate() {
            ret.data[i] = a * rhs;
        }
        ret
    }
}

impl<const S: usize> Mul<DenseSimd<S>> for f32 {
    type Output = DenseSimd<S>;

    fn mul(self, rhs: DenseSimd<S>) -> Self::Output {
        rhs * self
    }
}

impl<const S: usize> Div for DenseSimd<S> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, (a, b)) in self.data.into_iter().zip(rhs.data.into_iter()).enumerate() {
            ret.data[i] = a * b;
        }
        ret
    }
}

impl<const S: usize> Div<f32> for DenseSimd<S> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let mut ret = Self { data: [0.; S] };
        for (i, a) in self.data.into_iter().enumerate() {
            ret.data[i] = a / rhs;
        }
        ret
    }
}

impl<const S: usize> Div<DenseSimd<S>> for f32 {
    type Output = DenseSimd<S>;

    fn div(self, rhs: DenseSimd<S>) -> Self::Output {
        let mut ret = DenseSimd { data: [0.; S] };
        for (i, a) in rhs.data.into_iter().enumerate() {
            ret.data[i] = self / a;
        }
        ret
    }
}
