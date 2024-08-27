use crate::{dual::Dual, Trainable};

#[derive(Debug)]
pub struct Linear {
    // y = xa + b
    a: Dual<2>,
    b: Dual<2>,
}

impl Linear {
    pub fn new() -> Self {
        Self {
            a: Dual::new_param(0., 0),
            b: Dual::new_param(0., 1),
        }
    }
}

impl Trainable<2, 1, 1> for Linear {
    fn get_params(&self) -> [f32; 2] {
        [self.a.get_real(), self.b.get_real()]
    }

    fn set_params(&mut self, params: [f32; 2]) {
        self.a = Dual::new_param(params[0], 0);
        self.b = Dual::new_param(params[1], 1);
    }

    fn eval(&self, input: &[Dual<2>; 1]) -> [Dual<2>; 1] {
        let ret = self.b + &(self.a * &input[0]);

        [ret]
    }
}
