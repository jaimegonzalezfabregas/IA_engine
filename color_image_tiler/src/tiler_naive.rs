use std::ops::{Add, Div, Mul, Sub};

use std::fmt::Debug;

use ia_engine::dual::extended_arithmetic::ExtendedArithmetic;
use number_traits::Sqrt;

use crate::{PARTICLE_FREEDOM, TILE_BIAS, TILE_COUNT, TILE_COUNT_SQRT};

// [x,y, r, g, b]
pub fn tiler<
    N: Clone
        + From<f32>
        + Debug
        + PartialOrd<f32>
        + PartialOrd<N>
        + Add<N, Output = N>
        + Sub<N, Output = N>
        + Mul<N, Output = N>
        + Div<N, Output = N>
        + Add<f32, Output = N>
        + Sub<f32, Output = N>
        + Mul<f32, Output = N>
        + Div<f32, Output = N>
        + ExtendedArithmetic,
>(
    params: &[N; TILE_COUNT * 5],
    input: &[f32; 2],
    _: &(),
) -> [N; 3] {
    let grid_size = PARTICLE_FREEDOM / TILE_COUNT_SQRT as f32;

    let cells: Vec<_> = params.array_chunks::<5>().collect();

    let mut closest_d = N::from(1.);
    let mut closest_i = 0;
    let mut second_closest_d = N::from(1.);
    let mut second_closest_i = 0;


    for i in 0..TILE_COUNT {
        let base_x = (i / TILE_COUNT_SQRT) as f32;
        let base_y = (i % TILE_COUNT_SQRT) as f32;

        let [relative_x, relative_y, _, _, _] = cells[i];

        let x = relative_x.clone() * grid_size + base_x / TILE_COUNT_SQRT as f32;
        let y = relative_y.clone() * grid_size + base_y / TILE_COUNT_SQRT as f32;

        let x_d = x.clone() - input[0];
        let y_d = y.clone() - input[1];

        let d_2 = x_d.pow2() + y_d.pow2();

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

    let [_, _, closest_r, closest_g, closest_b] = cells[closest_i].clone();

    let mix_factor = closest_d / second_closest_d;

    if mix_factor < TILE_BIAS {
        [closest_r.clone(), closest_b.clone(), closest_g.clone()]
    } else {
        let [_, _, second_closest_r, second_closest_g, second_closest_b] =
            cells[second_closest_i].clone();

        let biased_mix_factor = if mix_factor < TILE_BIAS {
            N::from(0.)
        } else {
            (mix_factor - TILE_BIAS) / (1. - TILE_BIAS) / 2.
        };

        let anti_biased_mix_factor = N::from(1.) - biased_mix_factor.clone();

        [
            (closest_r * anti_biased_mix_factor.clone())
                + (second_closest_r * biased_mix_factor.clone()),
            (closest_g * anti_biased_mix_factor.clone())
                + (second_closest_g * biased_mix_factor.clone()),
            (closest_b * anti_biased_mix_factor) + (second_closest_b * biased_mix_factor),
        ]
    }
}
