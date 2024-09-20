use std::ops::{Add, Mul};

pub fn polinomial<const G: usize, N: Clone + From<f32> + PartialOrd<f32> + PartialOrd<N>>(
    params: &[N; G],
    input: &[N; 1],
    _: &(),
) -> [N; 1]
where
    for<'own, 'a, 'b, 'c, 'd> &'own N:
        Add<&'b N, Output = N> + Mul<&'c N, Output = N>
{
    let mut ret = N::from(0.);
    let mut x_to_the_nth = N::from(1.);

    for n in 0..G {
        ret = &ret + &(&x_to_the_nth * &params[n]);

        x_to_the_nth = &x_to_the_nth * &input[0];
    }

    [ret]
}
