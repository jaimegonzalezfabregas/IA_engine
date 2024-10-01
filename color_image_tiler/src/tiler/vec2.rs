use std::ops::{Add, Div, Mul, Sub};

use ia_engine::dual::extended_arithmetic::ExtendedArithmetic;

#[derive(Clone)]
pub struct Vec2<N> {
    pub x: N,
    pub y: N,
}

impl<N: Mul<M, Output = N>, M: Clone> Mul<M> for Vec2<N> {
    type Output = Vec2<N>;

    fn mul(self, rhs: M) -> Self::Output {
        Vec2 {
            x: self.x * rhs.clone(),
            y: self.y * rhs,
        }
    }
}

impl<N: Sub<Output = N>> Sub<Vec2<N>> for Vec2<N> {
    type Output = Vec2<N>;

    fn sub(self, rhs: Vec2<N>) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<N: From<f32> + ExtendedArithmetic + Add<N, Output = N>> Vec2<N> {
    pub fn zero() -> Self {
        Self {
            x: N::from(0.),
            y: N::from(0.),
        }
    }

    pub fn new(x: N, y: N) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> N {
        self.length2().sqrt()
    }

    pub fn length2(self) -> N {
        self.x.pow2() + self.y.pow2()
    }

}

pub fn dot<N: Clone + Add<N, Output = N> + Mul<N, Output = N>>(p: Vec2<N>, v: Vec2<N>) -> N {
    p.x * v.x + p.y * v.y
}

fn normalize<
    N: Clone + From<f32> + Add<N, Output = N> + Div<N, Output = N> + ExtendedArithmetic + PartialEq,
>(
    mut v: Vec2<N>,
) -> Vec2<N> {
    // Calculate the length (magnitude) of the vector
    let length = v.clone().length();

    // Check for zero length to avoid division by zero
    if length == N::from(0.0) {
        v
    } else {
        v.x = v.x.clone() / length.clone();
        v.y = v.y.clone() / length;
        v
    }
}

pub fn project_point_to_vector<
    N: Clone
        + From<f32>
        + PartialEq
        + Add<N, Output = N>
        + Sub<N, Output = N>
        + Mul<N, Output = N>
        + Div<N, Output = N>
        + ExtendedArithmetic,
>(
    p: Vec2<N>,
    v: Vec2<N>,
) -> Vec2<N> {
    // Normalize the vector V
    let normalized = normalize(v);

    // Calculate the projection length of P onto V
    let projection_length = dot(p, normalized.clone());

    // Calculate the projected point
    let projected_point = normalized * projection_length;

    return projected_point;
}
