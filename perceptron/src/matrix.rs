use std::array;
use std::ops::Add;
use std::ops::Mul;

use rand::Rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize, T> {
    data: [[T; COLS]; ROWS],
}
impl<const W: usize, const H: usize, N: From<f32>> Matrix<H, W, N> {
    pub fn delinearize<F: Fn(&mut N)>(&mut self, delinearizer: F) {
        for y in 0..H {
            for x in 0..W {
                delinearizer(&mut self.data[y][x]);
            }
        }
    }

    pub fn new(data: [[N; W]; H]) -> Self {
        Matrix { data }
    }

    pub fn cero() -> Self {
        let data = array::from_fn(|_| array::from_fn(|_| N::from(0.)));
        Matrix { data }
    }
}

impl<
        const W: usize,
        const H: usize,
        N: Mul<N, Output = N> + Add<N, Output = N> + From<f32> + Clone,
    > Matrix<H, W, N>
{
    pub fn from_flat(flat_data: &[N]) -> Self {
        assert_eq!(flat_data.len(), W * H);
        
        Matrix {
            data: array::from_fn(|y| array::from_fn(|x| flat_data[y * W * x].clone())),
        }
    }
}

impl<const H: usize, N: Clone> Matrix<H, 1, N> {
    pub fn as_array(self) -> [N; H] {
        array::from_fn(|y| self.data[y][1].clone())
    }
}

impl<const W: usize, const H: usize> Matrix<H, W, f32> {
    pub fn rand<R: Rng>(generator: &mut R) -> Self {
        let data = array::from_fn(|_| array::from_fn(|_| generator.gen()));
        Matrix { data }
    }
}

impl<const S: usize, N: Clone + Mul<N, Output = N> + Add<N, Output = N> + From<f32>>
    Matrix<S, S, N>
{
    pub fn unit() -> Self {
        let mut ret = Self::cero();
        for i in 0..S {
            ret.data[i][i] = N::from(1.0);
        }
        ret
    }

    pub fn hola(&self) -> usize {
        0
    }
}

impl<const L: usize, const M: usize, const N: usize, T> Mul<Matrix<M, L, T>> for Matrix<N, M, T>
where
    T: Mul<Output = T> + Add<Output = T> + From<f32> + Clone,
{
    type Output = Matrix<N, L, T>;

    fn mul(self, rhs: Matrix<M, L, T>) -> Self::Output {
        let mut result = Matrix::<N, L, T>::cero();

        for i in 0..N {
            for j in 0..L {
                for k in 0..M {
                    result.data[i][j] = result.data[i][j].clone()
                        + (self.data[i][k].clone() * rhs.data[k][j].clone());
                }
            }
        }

        result
    }
}

impl<const N: usize, const M: usize, T> Add for Matrix<N, M, T>
where
    T: Add<Output = T> + Clone + From<f32>,
{
    type Output = Matrix<N, M, T>;

    fn add(self, rhs: Matrix<N, M, T>) -> Self::Output {
        let mut result = Matrix::<N, M, T>::cero();

        for i in 0..N {
            for j in 0..M {
                result.data[i][j] = self.data[i][j].clone() + rhs.data[i][j].clone();
            }
        }

        result
    }
}

#[cfg(test)]
mod matrix_tests {

    use super::Matrix;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaChaRng;

    #[test]
    fn stress_test_multiplication() {
        let mut rand = ChaChaRng::seed_from_u64(0);

        for _ in 0..100 {
            let unit3 = Matrix::unit();
            let unit4 = Matrix::unit();
            let a: Matrix<3, 3, f32> = Matrix::rand(&mut rand);
            let b: Matrix<3, 3, f32> = Matrix::rand(&mut rand);

            assert_eq!(a, a.clone() * unit3.clone());
            assert_eq!(a, unit4 * a.clone());

            let ab = a.clone() * b.clone();

            assert_eq!(ab, a.clone() * (unit3.clone() * b.clone()));
            assert_eq!(ab, (a.clone() * unit3) * b.clone());
        }
    }

    #[test]
    fn test_multiplication_a() {
        let a = Matrix::new([[1., 2., 3.], [1., 0., 3.], [0., 2., 3.]]);
        let b = Matrix::new([[3., 2., 3.], [1., 3., 2.], [7., 6., 3.]]);

        let ab = Matrix::new([[26., 26., 16.], [24., 20., 12.], [23., 24., 13.]]);

        assert_eq!(a * b, ab);
    }

    #[test]
    fn test_multiplication_b() {
        let a = Matrix::new([[1., 2.], [3., 4.], [5., 6.]]);
        let b = Matrix::new([[7., 8., 9., 10.], [11., 12., 13., 14.]]);

        let ab = Matrix::new([
            [29., 32., 35., 38.],
            [65., 72., 79., 86.],
            [101., 112., 123., 134.],
        ]);

        assert_eq!(a * b, ab);
    }

    #[test]
    fn test_multiplication_papa() {
        let a = Matrix::new([[1., 2., 3.], [3., 4., 0.]]);
        let b = Matrix::new([[7., 8.], [7., 9.], [9., 8.]]);

        let ab = Matrix::new([[48., 50.], [49., 60.]]);

        assert_eq!(a * b, ab);
    }
}
