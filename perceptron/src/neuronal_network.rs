
use std::fmt::Debug;
use std::ops::{Add, Mul};

pub fn neuronal_network<
    N: Clone
        + Debug
        + From<f32>
        + PartialOrd<f32>
        + PartialOrd<N>
        + Add<N, Output = N>
        + Mul<N, Output = N>
        + Mul<f32, Output = N>,
>(
    params: &[N; 10],
    input: &[f32; 28*28],
    _: &(),
) -> [N; 10] {
    let mut ret = N::from(0.);
    let mut x_to_the_nth = N::from(1.);

    for n in 0..G {
      
        ret = ret + (x_to_the_nth.clone() * params[n].clone());

        x_to_the_nth = x_to_the_nth * input[0];
    }

    [ret]
}
