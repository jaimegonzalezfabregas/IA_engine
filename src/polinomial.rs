use std::array;

use crate::{dual::Dual, Trainable};

#[derive(Debug)]
pub struct Polinomial<const G: usize> {
    factors: [Dual<G>; G],
}

impl<const G: usize> Polinomial<G> {
    pub fn new() -> Self {
        Self {
            factors: array::from_fn(|i| Dual::new_param(0., i)),
        }
    }
}

impl<const G: usize> Trainable<G, 1, 1> for Polinomial<G> {
    fn get_params(&self) -> [f32; G] {
        array::from_fn(|i| self.factors[i].get_real())
    }

    fn set_params(&mut self, params: [f32; G]) {
        self.factors = array::from_fn(|i| Dual::new_param(params[i], i));
    }

    fn eval(&self, input: &[Dual<G>; 1]) -> [Dual<G>; 1] {
        let mut ret = Dual::cero();
        let mut x_to_the_nth = Dual::new(1.);

        for n in 0..G {
            ret += &(x_to_the_nth * &self.factors[n]);

            x_to_the_nth *= &input[0];
        }

        [ret]
    }
}
