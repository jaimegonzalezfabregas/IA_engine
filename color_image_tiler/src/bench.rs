
extern crate test;

#[cfg(test)]
mod tests {
    use std::thread;

    use crate::{training_thread::train_work, TILE_COUNT};

    use super::test::Bencher;

    use ia_engine::simd_arr::{dense_simd::DenseSimd, hybrid_simd::HybridSimd, SimdArr};

    fn bench<S: SimdArr<{ TILE_COUNT * 5 }>>() {
        let train_builder = thread::Builder::new()
            .name("train_thread".into())
            .stack_size(2 * 1024 * 1024 * 1024);

        let _ = train_builder
            .spawn(|| train_work::<S>(None, Some(100)))
            .unwrap()
            .join();
    }

    #[bench]
    fn bench_dense(b: &mut Bencher) {
        b.iter(|| bench::<DenseSimd<_>>());
    }
    #[bench]
    fn bench_hybrid_1(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 1>>());
    }
    #[bench]
    fn bench_hybrid_2(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 2>>());
    }
    #[bench]
    fn bench_hybrid_3(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 3>>());
    }
    #[bench]
    fn bench_hybrid_4(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 4>>());
    }
    #[bench]
    fn bench_hybrid_10(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 10>>());
    }
    #[bench]
    fn bench_hybrid_100(b: &mut Bencher) {
        b.iter(|| bench::<HybridSimd<_, 100>>());
    }
}
