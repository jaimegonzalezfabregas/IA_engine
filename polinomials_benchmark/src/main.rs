mod polinomial;

use ia_engine::trainer::{DataPoint, Trainer};
use crate::polinomial::polinomial;

fn base_func(x: f32) -> f32 {
    1. * (x * x * x * x * x) - 4. * (x * x * x * x) - 10. * (x * x * x)
        + 40. * (x * x)
        + 9. * x
        + -11.
}

const SPEED: isize = 100;

fn main() {
    

    let mut trainer = Trainer::new(polinomial::<6,_>, |x|*x, ());

    let mut epoch = 10;

    while epoch > SPEED * 4 {
        for _ in 0..10000000 {
            let done = trainer.train_step(&dataset_service(epoch));
            if !done {
                epoch += 1;
                break;
            }
        }
    }
}

fn dataset_service<const P: usize>(epoch: isize) -> Vec<DataPoint<P, 1, 1>> {
    let abs_max = epoch;
    (-abs_max..abs_max)
        .map(|x| x as f32 / SPEED as f32)
        // .map(|x| x as f32 / 10.)
        .map(|x| DataPoint {
            input: [x],
            output: [base_func(x)],
        })
        .collect::<Vec<_>>()
}
