use std::{array, ops::{Add, Div, Mul, Sub}};

use number_traits::{Sqrt, Zero};


// [x,y, r, g, b]
pub fn compresor<
    N: Copy
        + From<f32>
        + Sub<Output = N>
        + Add<Output = N>
        + Mul<Output = N>
        + Div<Output = N>
        + Sqrt
        + Zero
>(
    params: &[N; 50],
    input: &[N; 2],
) -> [N; 3] {

    let cells: Vec<_> = params.array_chunks::<5>().collect();
    let distances: [N; 10] = array::from_fn(|i|{
        let [x,y,_,_,_] = *cells[i]; 
         let relative_x = input[0] - x;
        let relative_y = input[1] - y;

        relative_x * relative_x + relative_y * relative_y

    });

    let distances_magnitude = distances.iter().fold(N::zero(), |acc, &d| acc + d);

    let normalized_distances : [N; 10] = array::from_fn(|i| distances[i] / distances_magnitude);

    normalized_distances.iter().enumerate().fold([N::zero(),N::zero(),N::zero()], |ret, (i, &d)| {
        let [_,_,r,g,b] = *cells[i]; 

        [ret[0] + r * d, ret[0] + g * d, ret[0] + b * d]
    })
}
