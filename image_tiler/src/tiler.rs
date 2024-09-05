use std::ops::{Add, Div, Mul, Neg, Sub};

use image::{DynamicImage, GenericImageView};
use number_traits::{One, Sqrt, Zero};
use std::fmt::Debug;

use crate::TILE_COUNT;

// [x,y, r, g, b]
pub fn tiler<
    N: Copy
        + From<f32>
        + Into<f32>
        + Sub<Output = N>
        + Add<Output = N>
        + Mul<Output = N>
        + Div<Output = N>
        + Sqrt
        + Zero
        + One
        + Debug
        + Neg<Output = N>
        + PartialOrd<f32>
        + PartialOrd<N>
        + Sub<f32, Output = N>
        + Div<f32, Output = N>
        + Add<f32, Output = N>,
>(
    params: &[N; TILE_COUNT * 2],
    input: &[N; 2],
    img: &DynamicImage,
) -> [N; 3] {
    const BIAS: f32 = 0.002;

    let cells: Vec<_> = params.array_chunks::<2>().collect();

    let mut closest_d = N::from(1.);
    let mut closest_i = 0;
    let mut second_closest_d = N::from(1.);
    let mut second_closest_i = 0;

    for i in 0..TILE_COUNT {
        let [x, y] = *cells[i];

        let x_d = x - input[0];
        let y_d = y - input[1];

        let d_2 = (x_d * x_d) + (y_d * y_d);

        if closest_d > d_2 {
            second_closest_d = closest_d;
            second_closest_i = closest_i;

            closest_d = d_2;
            closest_i = i;
        } else if second_closest_d > d_2 {
            second_closest_d = d_2;
            second_closest_i = i;
        }
    }

    let [closest_x, closest_y] = *cells[closest_i];
    let [closest_r, closest_g, closest_b, _] = img
            .get_pixel((closest_x.into() * 999.) as u32, (closest_y.into() * 999.) as u32)
        .0
        .map(|val| N::from(val as f32 / u8::MAX as f32));

    if closest_d < BIAS {
        [closest_r, closest_b, closest_g]
    } else {
        let [second_closest_x, second_closest_y] = *cells[second_closest_i];

        let [second_closest_r, second_closest_g, second_closest_b, _] = img
            .get_pixel((second_closest_x.into() * 999.) as u32, (second_closest_y.into() * 999.) as u32)
            .0
            .map(|val| N::from(val as f32 / u8::MAX as f32));

        let half_r = closest_r + second_closest_r / 2.;
        let half_g = closest_g + second_closest_g / 2.;
        let half_b = closest_b + second_closest_b / 2.;

        let mix_factor = closest_d / second_closest_d;

        let biased_mix_factor = if mix_factor < BIAS {
            N::zero()
        } else {
            (mix_factor - BIAS) / (1. - BIAS)
        };

        let anti_biased_mix_factor = -biased_mix_factor + 1.;

        [
            closest_r * anti_biased_mix_factor + half_r * biased_mix_factor,
            closest_g * anti_biased_mix_factor + half_g * biased_mix_factor,
            closest_b * anti_biased_mix_factor + half_b * biased_mix_factor,
        ]
    }
}