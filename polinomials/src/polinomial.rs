use std::ops::{Add, Mul};

pub fn polinomial<
    const G: usize,
    N: Clone + From<f32> + PartialOrd<f32> + PartialOrd<N> + Add<N, Output = N> + Mul<N, Output = N>,
>(
    params: &[N; G],
    input: &[f32; 1],
    _: &(),
) -> [N; 1] {
    let mut ret = N::from(0.);
    let mut x_to_the_nth = N::from(1.);

    for n in 0..G {
        ret = ret + (x_to_the_nth.clone() * params[n].clone());

        x_to_the_nth = x_to_the_nth * input[0];
    }

    [ret]
}
