use std::{array, ops::{Add, Div, Mul, Sub}};

use number_traits::{One, Sqrt, Zero};
use std::fmt::Debug;

use crate::TILE_COUNT;



// [x,y, r, g, b]
pub fn tiler<
    N: Copy
        + From<f32>
        + Sub<Output = N>
        + Add<Output = N>
        + Mul<Output = N>
        + Div<Output = N>
        + Sqrt
        + Zero
        + One
        + Debug
        
>(
    params: &[N; TILE_COUNT * 5],
    input: &[N; 2],
) -> [N; 3] {

    let cells: Vec<_> = params.array_chunks::<5>().collect();
    let distances: [N; TILE_COUNT] = array::from_fn(|i|{
        let [x,y,_,_,_] = *cells[i]; 
        let relative_x = input[0] - x;
        let relative_y = input[1] - y;

        let d_2 = N::from(0.1) + relative_x * relative_x + relative_y * relative_y;
        let d_4 = d_2 * d_2;
        let d_8 = d_4 * d_4;
        N::one() / (d_8 * d_8)
    });

    let distances_magnitude = distances.iter().fold(N::zero(), |acc, &d| acc + d);

    let normalized_distances : [N; TILE_COUNT] = array::from_fn(|i| distances[i] / distances_magnitude);

    let ret = normalized_distances.iter().enumerate().fold([N::zero(); 3], |ret, (i, &d)| {
        let [_,_,r,g,b] = *cells[i]; 


        [ret[0] + (r * d), ret[1] + (g * d), ret[2] + (b * d)]
    });


    ret

}