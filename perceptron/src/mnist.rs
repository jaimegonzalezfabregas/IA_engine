use std::{array, fs::File, io::Cursor};

use byteorder::{BigEndian, ReadBytesExt};
use ia_engine::trainer::DataPoint;
use std::io::Read;

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(mut f: File) -> Result<MnistData, std::io::Error> {
        let mut contents: Vec<u8> = Vec::new();
        f.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

pub fn load_data<const P: usize>(
    dataset_name: &str,
) -> Result<Vec<DataPoint<P, { 14 * 14 }, 10>>, std::io::Error> {
    let label_data = MnistData::new((File::open(format!("{}-labels-idx1-ubyte", dataset_name)))?)?;
    let images_data = MnistData::new((File::open(format!("{}-images-idx3-ubyte", dataset_name)))?)?;
    let mut images = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    assert_eq!(image_shape, 28 * 28);

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();

        let mut downsampled_image_data = vec![0.; 14 * 14];

        for (offset_x, offset_y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            for x in 0..14 {
                for y in 0..14 {
                    downsampled_image_data[y * 14 + x] +=
                        image_data[(y * 2 + offset_x) * 28 + (x * 2 + offset_y)] as f32 / 255. / 4.
                }
            }
        }

        // println!("{:?}, {:?}", downsampled_image_data, image_data);

        let image_data: [f32; 14 * 14] =
            array::from_fn(|i| downsampled_image_data[i]);
        images.push(image_data);
    }

    let classifications = label_data.data;

    let mut ret = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        let mut out = [0.; 10];
        out[classification as usize] = 1.;

        ret.push(DataPoint {
            input: image,
            output: out,
        })
    }

    Ok(ret)
}

#[cfg(test)]
mod mnist_tests {

    use super::load_data;

    #[test]
    fn load_t10k() {
        let dataset = load_data::<0>("mnist/t10k").unwrap();
        assert_eq!(dataset.len(), 10_000);
    }

    #[test]
    fn load_train() {
        let dataset = load_data::<0>("mnist/train").unwrap();
        assert_eq!(dataset.len(), 60_000);
    }
}
