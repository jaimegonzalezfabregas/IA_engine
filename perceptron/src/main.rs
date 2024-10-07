#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
extern crate image as im;
extern crate piston_window;
extern crate vecmath;

mod matrix;
mod mnist;
mod neuronal_network;

use std::{env, thread};

use ia_engine::trainer::{
    default_param_translator, param_translator_with_bounds, CriticalityCue, DataPoint, Trainer,
};
use im::imageops::FilterType::Nearest;
use math::Matrix2d;
use mnist::load_data;
use neuronal_network::neuronal_network;
use rand::seq::SliceRandom;

use piston_window::*;
use vecmath::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    match args[1].as_str() {
        "train" => thread::Builder::new()
            .stack_size(20_000_000_000)
            .name("big stack main".into())
            .spawn(train_main)
            .unwrap()
            .join()
            .unwrap(),
        "demo" => main_demo(),
        x => unimplemented!("{}", x),
    }
}

fn main_demo() {
    let opengl = OpenGL::V3_2;
    let (width, height) = (280, 280);
    let mut window: PistonWindow = WindowSettings::new("piston: paint", (width, height))
        .exit_on_esc(true)
        .graphics_api(opengl)
        .build()
        .unwrap();

    let mut canvas = im::ImageBuffer::new(28, 28);
    let mut draw = false;
    let mut texture_context = TextureContext {
        factory: window.factory.clone(),
        encoder: window.factory.create_command_buffer().into(),
    };

    let mut settings = TextureSettings::new();
    settings.set_filter(Filter::Nearest);

    let mut texture: G2dTexture =
        Texture::from_image(&mut texture_context, &canvas, &settings).unwrap();

    let mut last_pos: Option<[f64; 2]> = None;

    while let Some(e) = window.next() {
        if e.render_args().is_some() {
            texture.update(&mut texture_context, &canvas).unwrap();
            window.draw_2d(&e, |c, g, device| {
                // Update texture before rendering.
                texture_context.encoder.flush(device);

                clear([1.0; 4], g);
                image(&texture, c.transform.scale(10., 10.), g);
            });
        }
        if let Some(button) = e.press_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = true;
            }
        };
        if let Some(button) = e.release_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = false;
                last_pos = None
            }
        };
        if draw {
            if let Some(pos) = e.mouse_cursor_args() {
                let (x, y) = (pos[0] as f32, pos[1] as f32);

                if let Some(p) = last_pos {
                    let (last_x, last_y) = (p[0] as f32, p[1] as f32);
                    let distance = vec2_len(vec2_sub(p, pos)) as u32;

                    for i in 0..distance {
                        let diff_x = x - last_x;
                        let diff_y = y - last_y;
                        let delta = i as f32 / distance as f32;
                        let new_x = (last_x + (diff_x * delta)) as u32;
                        let new_y = (last_y + (diff_y * delta)) as u32;
                        if new_x < width && new_y < height {
                            canvas.put_pixel(new_x / 10, new_y / 10, im::Rgba([0, 0, 0, 255]));
                        };
                    }
                };

                last_pos = Some(pos)
            };
        }
    }
}

fn train_main() {
    let mut rng = rand::thread_rng();

    let mut dataset = load_data("mnist/t10k").unwrap();

    dataset.shuffle(&mut rng);

    let mut trainer = Trainer::new_heap_hybrid(
        CriticalityCue::<{ 63610 / 4 }>(),
        neuronal_network::<{ 28 * 28 }, 10, 63610, _>,
        neuronal_network::<{ 28 * 28 }, _, _, _>,
        default_param_translator,
        // param_translator_with_bounds::<_, 4, -4>,
        vec![28 * 28, 80, 10],
    );

    trainer.load("model.bin");

    while trainer.train_stocastic_step::<true, _>(&dataset, 16 * 16, |i, trainer| {
        println!("{} / {}", i * 16 * 16, dataset.len());
        trainer.save("model.bin").unwrap();
    }) {
        // println!("{:?}", trainer.get_model_params());
        println!("{:?}", trainer.get_last_cost());
        dataset.shuffle(&mut rng);
    }

    // for dp in dataset {
    //     let prediction = neuronal_network::<{ 28 * 28 }, 10, _, _>(
    //         &trainer.get_model_params(),
    //         &dp.input,
    //         &vec![28*28, 80, 10],
    //     );

    //     println!(
    //         "for input: {:?} got {:?} aproximating {:?}",
    //         dp.input,
    //         prediction[0],
    //         dp.output[0],
    //     );
    // }
}
