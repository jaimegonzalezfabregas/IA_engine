#![feature(array_chunks)]

mod tiler;
mod training_thread;

use std::{
    array,
    sync::mpsc::channel,
    thread,
    time::SystemTime,
};

use crate::tiler::tiler;

extern crate camera_controllers;
extern crate piston_window;
extern crate vecmath;
#[macro_use]
extern crate gfx;
extern crate shader_version;

use training_thread::train_thread;

//----------------------------------------
// Cube associated data

gfx_vertex_struct!(Vertex {
    a_pos: [i8; 4] = "a_pos",
    a_tex_coord: [i8; 2] = "a_tex_coord",
});

impl Vertex {
    fn new(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
        Vertex {
            a_pos: [pos[0], pos[1], pos[2], 1],
            a_tex_coord: tc,
        }
    }
}

const TILE_COUNT: usize = 30;

gfx_defines! {

    constant Globals {
        time: f32 = "u_time",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        u_time: gfx::ConstantBuffer<Globals> = "b_Globals",
        out_color: gfx::RenderTarget<::gfx::format::Srgba8> = "o_Color",
        t_point: gfx::TextureSampler<[f32; 4]> = "t_point",
        t_color: gfx::TextureSampler<[f32; 4]> = "t_color",
    }

}

fn as_u8(i: f32) -> u8 {
    (u8::MAX as f32 * i) as u8
}

fn main() {
    use gfx::traits::*;
    use piston_window::*;
    use shader_version::glsl::GLSL;
    use shader_version::Shaders;

    let opengl = OpenGL::V3_2;

    let mut window: PistonWindow = WindowSettings::new("piston: cube", [640, 640])
        .exit_on_esc(true)
        .samples(4)
        .graphics_api(opengl)
        .build()
        .unwrap();

    let mut factory = window.factory.clone();

    let vertex_data = vec![
        //top (0, 0, 1)
        Vertex::new([-1, -1, -1], [0, 0]),
        Vertex::new([1, -1, -1], [1, 0]),
        Vertex::new([1, 1, -1], [1, 1]),
        Vertex::new([-1, 1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
    ];

    let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, index_data);

    let glsl = opengl.to_glsl();
    let pso = factory
        .create_pipeline_simple(
            Shaders::new()
                .set(
                    GLSL::V1_50,
                    &include_str!("../assets/cube_150.glslv")
                        .replace("TILE_COUNT", &format!("{TILE_COUNT}")),
                )
                .get(glsl)
                .unwrap()
                .as_bytes(),
            Shaders::new()
                .set(
                    GLSL::V1_50,
                    &include_str!("../assets/cube_150.glslf")
                        .replace("TILE_COUNT", &format!("{TILE_COUNT}")),
                )
                .get(glsl)
                .unwrap()
                .as_bytes(),
            pipe::new(),
        )
        .unwrap();

    let (tx, rx) = channel();

    let builder = thread::Builder::new()
        .name("train_thread".into())
        .stack_size(20 * 1024 * 1024 * 1024);

    builder.spawn(|| train_thread(tx)).unwrap();

    let mut params = rx.recv().unwrap();

    let now = SystemTime::now();
    while let Some(e) = window.next() {
        while let Ok(new_params) = rx.try_recv() {
            params = new_params;
        }

        let t = now.elapsed().unwrap().as_millis() as f32 / 1000.;

        let points: [[u8; 4]; TILE_COUNT] =
            array::from_fn(|i| [as_u8(params[i * 5]), as_u8(params[i * 5 + 1]), 0, 0]);
        let colors: [[u8; 4]; TILE_COUNT] = array::from_fn(|i| {
            [
                as_u8(params[i * 5 + 2]),
                as_u8(params[i * 5 + 3]),
                as_u8(params[i * 5 + 4]),
                u8::MAX,
            ]
        });

        let (_, texture_view_points) = factory
            .create_texture_immutable::<gfx::format::Rgba8>(
                gfx::texture::Kind::D2(TILE_COUNT as u16, 1, gfx::texture::AaMode::Single),
                gfx::texture::Mipmap::Provided,
                &[&points],
            )
            .unwrap();

        let (_, texture_view_colors) = factory
            .create_texture_immutable::<gfx::format::Rgba8>(
                gfx::texture::Kind::D2(TILE_COUNT as u16, 1, gfx::texture::AaMode::Single),
                gfx::texture::Mipmap::Provided,
                &[&colors],
            )
            .unwrap();

        let sinfo = gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Scale,
            gfx::texture::WrapMode::Clamp,
        );

        let data = pipe::Data {
            vbuf: vbuf.clone(),
            out_color: window.output_color.clone(),
            u_time: factory.create_constant_buffer(1),
            t_color: (texture_view_colors, factory.create_sampler(sinfo)),
            t_point: (texture_view_points, factory.create_sampler(sinfo)),
        };

        window.draw_3d(&e, |window| {
            window
                .encoder
                .update_constant_buffer(&data.u_time, &Globals { time: t % 1. });

            window
                .encoder
                .clear(&window.output_color, [0., 0., t % 1., 1.0]);

            window.encoder.draw(&slice, &pso, &data);
        });
    }
}
