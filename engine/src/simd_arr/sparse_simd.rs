use std::{
    array,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

use super::{DereferenceArithmetic, SimdArr};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseSimd<const S: usize> {
    data_index: [usize; S],
    data: [f32; S],
    size: usize,
}
impl<const S: usize> SparseSimd<S> {
    pub const fn non_zero_count(&self) -> usize {
        self.size
    }
}

impl<const S: usize> DereferenceArithmetic<SparseSimd<S>> for &SparseSimd<S> {}

impl<const S: usize> SimdArr<S> for SparseSimd<S> {
    fn zero() -> SparseSimd<S> {
        Self {
            data_index: [S; S],
            data: [0.; S],
            size: 0,
        }
    }

    fn new_from_array(arr: &[f32; S]) -> Self {
        let mut cursor = 0;
        let mut ret = Self::zero();
        for (i, &elm) in arr.iter().enumerate() {
            if elm != 0. {
                ret.data_index[cursor] = i;
                ret.data[cursor] = elm;
                cursor += 1;
            }
        }
        ret.size = cursor;
        ret
    }

    fn to_array(&self) -> [f32; S] {
        let mut cursor = 0;

        array::from_fn(|i| {
            if cursor == self.size {
                0.
            } else {
                if self.data_index[cursor] > i {
                    0.
                } else if self.data_index[cursor] == i {
                    cursor += 1;
                    self.data[cursor - 1]
                } else {
                    unreachable!()
                }
            }
        })
    }

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self {
        let mut ret = Self {
            data_index: [S; S],
            data: [0.; S],
            size: 1,
        };
        ret.data_index[0] = pos;
        ret.data[0] = val;
        ret
    }

    fn neg(&mut self) {
        for i in 0..self.size {
            self.data[i] *= -1.;
        }
    }
}

impl<const S: usize> Index<usize> for SparseSimd<S> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        let dereference_index = self.data_index.partition_point(|i| *i < index);
        if self.data_index[dereference_index] == index {
            &self.data[dereference_index]
        } else {
            &0.
        }
    }
}

impl<const S: usize> IndexMut<usize> for SparseSimd<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let dereference_index = self.data_index.partition_point(|i| *i < index);

        // println!("{index} {dereference_index}, {self:?}");

        if self.data_index[dereference_index] == index {
            &mut self.data[dereference_index]
        } else {
            self.size += 1;
            self.data_index[dereference_index..self.size].rotate_right(1);
            self.data[dereference_index..self.size].rotate_right(1);

            self.data_index[dereference_index] = index;
            &mut self.data[dereference_index]
        }
    }
}

impl<const S: usize> Add<&SparseSimd<S>> for SparseSimd<S> {
    type Output = SparseSimd<S>;

    fn add(self, rhs: &SparseSimd<S>) -> Self::Output {
        &self + rhs
    }
}

impl<const S: usize> Sub<&SparseSimd<S>> for SparseSimd<S> {
    type Output = SparseSimd<S>;

    fn sub(self, rhs: &SparseSimd<S>) -> Self::Output {
        &self - rhs
    }
}

impl<const S: usize> Add<&SparseSimd<S>> for &SparseSimd<S> {
    type Output = SparseSimd<S>;

    fn add(self, rhs: &SparseSimd<S>) -> Self::Output {
        let mut ret = SparseSimd {
            data_index: [S; S],
            data: [0.; S],
            size: 0,
        };

        let mut rhs_cursor = 0;
        let mut ret_cursor = 0;

        for self_cursor in 0..self.size {
            while rhs.data_index[rhs_cursor] < self.data_index[self_cursor] {
                ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
                ret.data[ret_cursor] = rhs.data[rhs_cursor];
                ret_cursor += 1;

                rhs_cursor += 1;
            }
            if rhs.data_index[rhs_cursor] == self.data_index[self_cursor] {
                ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
                ret.data[ret_cursor] = self.data[self_cursor] + rhs.data[rhs_cursor];
                ret_cursor += 1;

                rhs_cursor += 1;
            } else {
                ret.data_index[ret_cursor] = self.data_index[self_cursor];
                ret.data[ret_cursor] = self.data[self_cursor];
                ret_cursor += 1;
            }
        }
        while rhs_cursor < rhs.size {
            ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
            ret.data[ret_cursor] = rhs.data[rhs_cursor];
            ret_cursor += 1;

            rhs_cursor += 1;
        }

        ret.size = ret_cursor;

        ret
    }
}

impl<const S: usize> Sub<&SparseSimd<S>> for &SparseSimd<S> {
    type Output = SparseSimd<S>;

    fn sub(self, rhs: &SparseSimd<S>) -> Self::Output {
        let mut ret = SparseSimd {
            data_index: [S; S],
            data: [0.; S],
            size: 0,
        };

        let mut rhs_cursor = 0;
        let mut ret_cursor = 0;

        for self_cursor in 0..self.size {
            while rhs.data_index[rhs_cursor] < self.data_index[self_cursor] {
                ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
                ret.data[ret_cursor] = -rhs.data[rhs_cursor];
                ret_cursor += 1;
                rhs_cursor += 1;
            }
            if rhs.data_index[rhs_cursor] == self.data_index[self_cursor] {
                ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
                ret.data[ret_cursor] = self.data[self_cursor] - rhs.data[rhs_cursor];
                ret_cursor += 1;
                rhs_cursor += 1;
            } else {
                ret.data_index[ret_cursor] = self.data_index[self_cursor];
                ret.data[ret_cursor] = self.data[self_cursor];
                ret_cursor += 1;
            }
        }
        while rhs_cursor < rhs.size {
            ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
            ret.data[ret_cursor] = -rhs.data[rhs_cursor];
            ret_cursor += 1;
            rhs_cursor += 1;
        }

        ret.size = ret_cursor;

        ret
    }
}

impl<const S: usize> Mul<f32> for &SparseSimd<S> {
    type Output = SparseSimd<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut ret = *self;

        for i in 0..self.size {
            ret.data[i] = self.data[i] * rhs;
        }
        ret
    }
}

impl<const S: usize> Div<f32> for &SparseSimd<S> {
    type Output = SparseSimd<S>;

    fn div(self, rhs: f32) -> Self::Output {
        let mut ret = *self;

        for i in 0..self.size {
            ret.data[i] = self.data[i] / rhs;
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use rand::{self, seq::SliceRandom, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::array::from_fn;

    use crate::simd_arr::{sparse_simd::SparseSimd, SimdArr};

    #[test]
    fn test_create() {
        assert_eq!(
            SparseSimd::new_from_array(&[1., 4., 2., 4.]),
            SparseSimd {
                data_index: [0, 1, 2, 3],
                data: [1., 4., 2., 4.],
                size: 4
            }
        );

        assert_eq!(
            SparseSimd::new_from_array(&[1., 4., 0., 0., 0., 2., 4.]),
            SparseSimd {
                data_index: [0, 1, 5, 6, 7, 7, 7],
                data: [1., 4., 2., 4., 0., 0., 0.],
                size: 4
            }
        );
    }

    fn test_add<const N: usize>(a: [f32; N], b: [f32; N]) {
        let res_vec = a
            .iter()
            .zip(b)
            .map(|(a_elm, b_elm)| a_elm + b_elm)
            .collect::<Vec<_>>();
        let res: [f32; N] = from_fn(|i| res_vec[i]);

        assert_eq!(
            (SparseSimd::new_from_array(&a) + &SparseSimd::new_from_array(&b)).to_array(),
            res
        )
    }

    fn test_sub<const N: usize>(a: [f32; N], b: [f32; N]) {
        let res_vec = a
            .iter()
            .zip(b)
            .map(|(a_elm, b_elm)| a_elm - b_elm)
            .collect::<Vec<_>>();
        let res: [f32; N] = from_fn(|i| res_vec[i]);

        assert_eq!(
            (SparseSimd::new_from_array(&a) - &SparseSimd::new_from_array(&b)).to_array(),
            res
        )
    }

    #[test]
    fn arithmetic_stress_test() {
        for _ in 0..1000 {
            let cero_ratio: f32 = rand::random();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));
            let b: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));

            test_add(a, b);
            test_sub(a, b);

            assert_eq!(a, SparseSimd::new_from_array(&a).to_array())
        }
    }

    #[test]
    fn consistency_stress_test() {
        for _ in 0..1000 {
            let cero_ratio: f32 = rand::random();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));

            assert_eq!(a, SparseSimd::new_from_array(&a).to_array())
        }
    }

    fn test_mul_scalar<const N: usize>(a: [f32; N], b: f32) {
        let res = a.map(|a_elm| a_elm * b);

        assert_eq!((&SparseSimd::new_from_array(&a) * b).to_array(), res)
    }

    fn test_div_scalar<const N: usize>(a: [f32; N], b: f32) {
        let res = a.map(|a_elm| a_elm / b);

        assert_eq!((&SparseSimd::new_from_array(&a) / b).to_array(), res)
    }

    #[test]
    fn stress_test_scalar() {
        for _ in 0..1000 {
            let cero_ratio: f32 = rand::random();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));
            let b = rand::random();
            test_mul_scalar(a, b);
            test_div_scalar(a, b);
        }
    }

    #[test]
    fn test_indexing() {
        for i in 0..10 {
            let mut test_arr = [0.; 10];
            test_arr[i] = 1.;
            let x = SparseSimd::new_from_array(&test_arr);

            for j in 0..10 {
                if i == j {
                    assert_eq!(1., x[j]);
                    assert_eq!(1., x.to_array()[j]);
                } else {
                    assert_eq!(0., x[j]);
                    assert_eq!(0., x.to_array()[j]);
                }
            }
        }
    }

    #[test]
    fn stress_test_to_array() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..1000 {
            let cero_ratio: f32 = rand::random();
            let a: [f32; 4] = rand::random::<[f32; 4]>().map(|x| (x - cero_ratio).max(0.));

            let mut x: SparseSimd<4> = SparseSimd::zero();
            let mut order = (0..4).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            for i in order {
                x[i] = a[i];
                println!("{:?} -> {:?}", x, x.to_array());
                assert_eq!(x.to_array()[i], a[i]);
            }

            for i in 0..4 {
                assert_eq!(x.to_array()[i], a[i]);
            }
        }
    }

    #[test]
    fn stress_test_to_array_overriding() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..1000 {
            let cero_ratio: f32 = rand::random();
            let a: [f32; 4] = rand::random::<[f32; 4]>().map(|x| (x - cero_ratio).max(0.));
            let b: [f32; 4] = rand::random::<[f32; 4]>().map(|x| (x - cero_ratio).max(0.));

            let mut y: SparseSimd<4> = SparseSimd::new_from_array(&b);
            let mut order = (0..4).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            for i in order {
                y[i] = a[i];
                assert_eq!(y.to_array()[i], a[i]);
            }

            for i in 0..4 {
                assert_eq!(y.to_array()[i], a[i]);
            }
        }
    }

    #[test]
    fn stress_test_indexing() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..1000 {
            let cero_ratio: f32 = rand::random();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));
            let b: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));

            let mut x: SparseSimd<32> = SparseSimd::zero();
            let mut y: SparseSimd<32> = SparseSimd::new_from_array(&b);
            let mut order = (0..32).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            for i in order {
                x[i] = a[i];
                y[i] = a[i];
            }

            for i in 0..32 {
                assert_eq!(x[i], a[i]);
                assert_eq!(y[i], a[i]);
            }
        }
    }

    #[test]
    fn test_indexing_mut() {
        for i in 0..10 {
            let mut x = SparseSimd::new_from_array(&[0.; 10]);
            x[i] = 1.;

            for j in 0..10 {
                if i == j {
                    assert_eq!(1., x[j]);
                } else {
                    assert_eq!(0., x[j]);
                }
            }
        }
    }
}
