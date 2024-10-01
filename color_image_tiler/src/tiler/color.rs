use std::ops::{Add, Mul, Sub};

#[derive(Clone)]
pub struct Color<N> {
    pub r: N,
    pub g: N,
    pub b: N,
}
impl<N> Color<N> {
    pub(crate) fn into_array(self) -> [N; 3] {
        [self.r, self.g, self.b]
    }
}

impl<N: Mul<N, Output = N> + Clone> Mul<N> for Color<N> {
    type Output = Color<N>;

    fn mul(self, rhs: N) -> Self::Output {
        Color {
            r: self.r * rhs.clone(),
            g: self.g * rhs.clone(),
            b: self.b * rhs,
        }
    }
}

impl<N: Add<N, Output = N>> Add<Color<N>> for Color<N> {
    type Output = Color<N>;

    fn add(self, rhs: Color<N>) -> Self::Output {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

pub fn get_color<N: Clone>(tile: &[N; 5]) -> Color<N> {
    Color {
        r: tile[2].clone(),
        g: tile[3].clone(),
        b: tile[4].clone(),
    }
}

pub fn mix<N: From<f32> + Sub<N, Output = N> + Mul<N, Output = N> + Add<N, Output = N> + Clone>(
    a: Color<N>,
    b: Color<N>,
    factor: N,
) -> Color<N> {
    let anti_factor = N::from(1.) - factor.clone();

    a * factor + b * anti_factor
}
