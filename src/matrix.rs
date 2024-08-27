use std::ops::Mul;

struct MatrixShape{
    cols: usize,
    rows: usize
}

impl MatrixShape{
    pub fn size(&self) -> usize{
        self.cols * self.rows
    }

    pub fn new(cols: usize, rows:usize)->Self{
        Self {
            cols,
            rows
        }
    }
}

struct Matrix<T>{
    elements: Vec<T>,d55la
    shape: MatrixShape
}

impl<T: Copy> Matrix<T>{
    fn get(&self, x:usize, y:usize) -> T{
        return self.elements[x + y * self.shape.cols]
    }
}

impl<T: Default + Clone> Matrix<T>{
    fn new(shape: MatrixShape) -> Self {
        Self { elements: vec![T::default(); shape.size()], shape }
    }

    fn map(&mut self, f: fn(T) -> T) {
        self.elements.iter_mut().map(|e| {*e = f(e);});
    }
}

impl<T: Default + Clone> Mul<Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;

    /*
        https://en.wikipedia.org/wiki/File:Matrix_multiplication_qtl1.svg
    */

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        let ret_shape = MatrixShape::new(rhs.shape.cols, self.shape.rows);

        let ret = Matrix::new(ret_shape);

        for x in 0..rhs.shape.cols{
            for y in 0..self.shape.rows{
                ret.set(x,y,)
            }
        }

        ret
    }
}