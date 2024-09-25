use std::ops::{Add, Div, Mul, Sub};

use std::fmt::Debug;

use ia_engine::dual::extended_arithmetic::ExtendedArithmetic;

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
    let cells: Vec<_> = params.array_chunks::<5>().collect();

    let grid_size = PARTICLE_FREEDOM as f32 / TILE_COUNT_SQRT as f32;

    let mut closest_d = N::from(1.);
    let mut closest_i = 0;
    let mut second_closest_d = N::from(1.);
    let mut second_closest_i = 0;

    let sample_grid_x = (input[0] * TILE_COUNT_SQRT as f32).floor();
    let sample_grid_y = (input[1] * TILE_COUNT_SQRT as f32).floor();

    for cell_dx in -PARTICLE_FREEDOM..=PARTICLE_FREEDOM {
        for cell_dy in -PARTICLE_FREEDOM..=PARTICLE_FREEDOM {
            let cell_x = sample_grid_x as isize + cell_dx;
            let cell_y = sample_grid_y as isize + cell_dy;

            if cell_x < 0
                || cell_y < 0
                || cell_x >= TILE_COUNT_SQRT as isize
                || cell_y >= TILE_COUNT_SQRT as isize
            {
                continue;
            }
            let i = (cell_x * TILE_COUNT_SQRT as isize + cell_y) as usize;

            let [relative_x, relative_y, _, _, _] = cells[i];

            let seed_x = relative_x.clone() * grid_size + cell_x as f32 / TILE_COUNT_SQRT as f32;
            let seed_y = relative_y.clone() * grid_size + cell_y as f32 / TILE_COUNT_SQRT as f32;

            let x_d = seed_x - input[0];
            let y_d = seed_y - input[1];

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
    }

    let [_, _, closest_r, closest_g, closest_b] = cells[closest_i].clone();

    if second_closest_d == 0. {
        [closest_r.clone(), closest_g.clone(), closest_b.clone()]
    } else {
        let mix_factor = closest_d / second_closest_d;

        if mix_factor <= TILE_BIAS {
            [closest_r.clone(), closest_g.clone(), closest_b.clone()]
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
}
