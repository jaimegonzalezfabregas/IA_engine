
pub fn polinomial<
    const G: usize,
    N: Copy
        + From<f32>
        + std::ops::MulAssign<N>
        + std::ops::Mul<N, Output = N>
        + std::ops::AddAssign<N>,
>(
    params: &[N; G],
    input: &[N; 1],
    _: &()
) -> [N; 1] {
    let mut ret = N::from(0.);
    let mut x_to_the_nth = N::from(1.);

    for n in 0..G {
        ret += x_to_the_nth * params[n];

        x_to_the_nth *= input[0];
    }

    [ret]
}
