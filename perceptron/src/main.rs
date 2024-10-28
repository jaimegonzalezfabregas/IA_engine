#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
extern crate image as im;
extern crate piston_window;
extern crate vecmath;

mod eval_thread;
mod matrix;
mod mnist;
mod neuronal_network;

use std::{env, thread, time::Instant};

use crate::eval_thread::eval_thread;
use ia_engine::trainer::{default_param_translator, CriticalityCue, Trainer};

use mnist::load_data;
use neuronal_network::neuronal_network;
use piston_window::*;
use rand::seq::SliceRandom;
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
    let mut last_change_time = None;
    let mut rng = rand::thread_rng();

    let mut dataset: Vec<ia_engine::trainer::DataPoint<0, _, 10>> =
        load_data("mnist/t10k").unwrap();

    let mut pixel_input = [0.; { 14 * 14 }];

    let opengl = OpenGL::V3_2;
    let (width, height) = (140, 140);
    let mut window: PistonWindow = WindowSettings::new("piston: paint", (width, height))
        .exit_on_esc(true)
        .graphics_api(opengl)
        .build()
        .unwrap();

    let mut canvas = im::ImageBuffer::new(14, 14);
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

    let mut change = false;

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

            if button == Button::Keyboard(Key::C) {
                pixel_input = [0.; 14 * 14];
                change = true;
            }

            if button == Button::Keyboard(Key::S) {
                dataset.shuffle(&mut rng);

                pixel_input = dataset[0].input;

                change = true;
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
                            pixel_input[(new_x / 10 + new_y / 10 * 14) as usize] = 1.;
                            change = true;
                        };
                    }
                };

                last_pos = Some(pos)
            };
        }

        if change {
            change = false;
            last_change_time = Some(Instant::now());

            for x in 0..14 {
                for y in 0..14 {
                    let c = 1. - pixel_input[(x + y * 14) as usize];
                    canvas.put_pixel(
                        x,
                        y,
                        im::Rgba([(c * 255.) as u8, (c * 255.) as u8, (c * 255.) as u8, 255]),
                    );
                }
            }
        }

        if let Some(t) = last_change_time {
            if t.elapsed().as_secs_f64() > 0.5 {
                last_change_time = None;
                let predition = thread::Builder::new()
                    .stack_size(20_000_000_000)
                    .name("big stack main".into())
                    .spawn(move || eval_thread(&pixel_input))
                    .unwrap()
                    .join()
                    .unwrap();

                let mut max_pos = 0;

                for (i, pred) in predition.iter().enumerate() {
                    if predition[max_pos] < *pred {
                        max_pos = i;
                    }
                }

                println!("predition: {max_pos}, {:?} ", predition)
            }
        }
    }
}

fn train_main() {
    let mut rng = rand::thread_rng();

    // let mut dataset = load_data("mnist/t10k").unwrap();
    let mut dataset = load_data("mnist/train").unwrap();

    let mut trainer = Trainer::new_hybrid(
        CriticalityCue::<{ 6740 / 20 }>(),
        neuronal_network::<{ 14 * 14 }, 10, 6740, _>,
        neuronal_network::<{ 14 * 14 }, _, _, _>,
        default_param_translator,
        // param_translator_with_bounds::<_, 4, -4>,
        vec![14 * 14, 30, 20, 10],
    );

    trainer.load("model.bin").unwrap();

    const SUBDATASET_SIZE: usize = 16 * 16 * 16;

    while trainer.train_stocastic_step::<true, _>(&dataset, SUBDATASET_SIZE, |i, trainer| {
        println!("{} / {}", i * SUBDATASET_SIZE, dataset.len());
        trainer.save("model.bin").unwrap();
        // trainer.shake(0.001);
        // println!("{:?}", trainer.get_last_cost());
    }) {
        // println!("{:?}", trainer.get_model_params());
        // println!("{:?}", trainer.get_last_cost());
        dataset.shuffle(&mut rng);
    }
}
