use std::{
    array,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SparseSimd<const S: usize> {
    data_index: [usize; S],
    data: [f32; S],
    size: usize,
}
impl<const S: usize> SparseSimd<S> {
    pub(crate) const fn zero() -> SparseSimd<S> {
        Self {
            data_index: [S; S],
            data: [0.; S],
            size: 0,
        }
    }

    pub(crate) const fn non_zero_count(&self) -> usize {
        self.size
    }

    pub(crate) fn new_from_array(arr: &[f32; S]) -> Self {
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

    pub(crate) fn to_array(&self) -> [f32; S] {
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

    pub(crate) fn merge_over_fn<F: Fn(f32, f32) -> f32>(&self, rhs: &Self, merger: F) -> Self {
        let mut ret = Self {
            data_index: [S; S],
            data: [0.; S],
            size: 0,
        };

        let mut rhs_cursor = 0;
        let mut ret_cursor = 0;

        for self_cursor in 0..self.size {
            while rhs.data_index[rhs_cursor] < self.data_index[self_cursor] {
                let potencial_value = merger(0., rhs.data[rhs_cursor]);
                if potencial_value != 0. {
                    ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
                    ret.data[ret_cursor] = potencial_value;
                    ret_cursor += 1;
                }
                rhs_cursor += 1;
            }
            if rhs.data_index[rhs_cursor] == self.data_index[self_cursor] {
                let potencial_value = merger(self.data[self_cursor], rhs.data[rhs_cursor]);
                if potencial_value != 0. {
                    ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
                    ret.data[ret_cursor] = potencial_value;
                    ret_cursor += 1;
                }
                rhs_cursor += 1;
            } else {
                let potencial_value = merger(self.data[self_cursor], 0.);

                if potencial_value != 0. {
                    ret.data_index[ret_cursor] = self.data_index[self_cursor];
                    ret.data[ret_cursor] = merger(self.data[self_cursor], 0.);
                    ret_cursor += 1;
                }
            }
        }
        while rhs_cursor < rhs.size {
            let potencial_value = merger(0., rhs.data[rhs_cursor]);

            if potencial_value != 0. {
                ret.data_index[ret_cursor] = rhs.data_index[rhs_cursor];
                ret.data[ret_cursor] = merger(0., rhs.data[rhs_cursor]);
                ret_cursor += 1;
            }
            rhs_cursor += 1;
        }

        ret.size = ret_cursor;

        ret
    }
}

impl<const S: usize> Index<usize> for SparseSimd<S> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        let dereference_index = self.data_index.partition_point(|i| *i >= index);
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
            self.data_index[dereference_index..].rotate_right(1);
            self.data[dereference_index..].rotate_right(1);

            self.data_index[dereference_index] = index;
            &mut self.data[dereference_index]
        }
    }
}

impl<const S: usize> Add for SparseSimd<S> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.merge_over_fn(&rhs, |a, b| a + b)
    }
}

impl<const S: usize> Sub for SparseSimd<S> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.merge_over_fn(&rhs, |a, b| a - b)
    }
}

impl<const S: usize> Mul for SparseSimd<S> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.merge_over_fn(&rhs, |a, b| a * b)
    }
}

impl<const S: usize> Mul<f32> for SparseSimd<S> {
    type Output = Self;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.data = self.data.map(|x| x * rhs);
        self
    }
}

impl<const S: usize> Div<f32> for SparseSimd<S> {
    type Output = Self;

    fn div(mut self, rhs: f32) -> Self::Output {
        self.data = self.data.map(|x| x / rhs);
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::sparse_simd::SparseSimd;
    use rand;
    use rayon::array;
    use std::array::from_fn;

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
            SparseSimd::new_from_array(&a) + SparseSimd::new_from_array(&b),
            SparseSimd::new_from_array(&res)
        )
    }

    fn test_mul<const N: usize>(a: [f32; N], b: [f32; N]) {
        let res_vec = a
            .iter()
            .zip(b)
            .map(|(a_elm, b_elm)| a_elm * b_elm)
            .collect::<Vec<_>>();
        let res: [f32; N] = from_fn(|i| res_vec[i]);

        assert_eq!(
            SparseSimd::new_from_array(&a) * SparseSimd::new_from_array(&b),
            SparseSimd::new_from_array(&res)
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
            SparseSimd::new_from_array(&a) - SparseSimd::new_from_array(&b),
            SparseSimd::new_from_array(&res)
        )
    }

    #[test]
    fn stress_test() {
        for _ in 0..100000 {
            let cero_ratio: f32 = rand::random();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));
            let b: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));

            test_add(a, b);
            test_mul(a, b);
            test_sub(a, b);
        }
    }
}
