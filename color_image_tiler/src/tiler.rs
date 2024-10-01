mod color;
mod vec2;

use std::ops::{Add, Div, Mul, Sub};

use std::fmt::Debug;

use color::{get_color, mix};
use ia_engine::dual::extended_arithmetic::ExtendedArithmetic;
use vec2::{project_point_to_vector, Vec2};

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
    let cells: Vec<_> = params.array_chunks::<5>().cloned().collect();

    let input_point = Vec2::new(N::from(input[0]), N::from(input[1]));

    let grid_size = PARTICLE_FREEDOM as f32 / TILE_COUNT_SQRT as f32;

    let mut closest_d = N::from(1.);
    let mut closest_i = 0;
    let mut closest_point = Vec2::zero();
    let mut second_closest_d = N::from(1.);
    let mut second_closest_i = 0;
    let mut second_closest_point = Vec2::zero();

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

            let [relative_x, relative_y, _, _, _] = &cells[i];

            let seed_point = Vec2::new(
                relative_x.clone() * grid_size + cell_x as f32 / TILE_COUNT_SQRT as f32,
                relative_y.clone() * grid_size + cell_y as f32 / TILE_COUNT_SQRT as f32,
            );

            let d = (seed_point.clone() - input_point.clone()).length2();

            if closest_d > d {
                second_closest_d = closest_d;
                second_closest_i = closest_i;
                second_closest_point = closest_point;

                closest_d = d;
                closest_i = i;
                closest_point = seed_point;
            } else if second_closest_d > d {
                second_closest_d = d;
                second_closest_i = i;
                second_closest_point = seed_point;
            }
        }
    }

    let closest_col = get_color(&cells[closest_i]);
    let second_closest_col = get_color(&cells[second_closest_i]);

    let gradient_direction = closest_point.clone() - second_closest_point.clone();

    let projected_closest_p = project_point_to_vector(closest_point, gradient_direction.clone());
    let projected_second_closest_p =
        project_point_to_vector(second_closest_point, gradient_direction.clone());
    let projected_input = project_point_to_vector(input_point, gradient_direction);

    let projected_closest_d = (projected_closest_p - projected_input.clone()).length();
    let projected_second_closest_d = (projected_second_closest_p - projected_input).length();

    let factor = projected_closest_d / projected_second_closest_d;

    if factor < TILE_BIAS {
        closest_col.into_array()
    } else {
        let factor = (factor - TILE_BIAS) / (N::from(1.) - TILE_BIAS);
        mix(
            closest_col.clone(),
            mix(closest_col, second_closest_col, N::from(0.5)),
            factor,
        )
        .into_array()
    }
}
