mod polinomial;

use full_palette::{GREEN_A700};
use piston_backend::draw_piston_window;
use piston_window::{PistonWindow, WindowSettings};
use plotters::prelude::*;
use polinomial::Polinomial;
use trainer::{DataPoint, Trainer};

fn polinomial(x: f32) -> f32 {
    1. * (x * x * x * x * x) - 4. * (x * x * x * x) - 10. * (x * x * x)
        + 40. * (x * x)
        + 9. * x
        + -11.
}

const SPEED: isize = 100;

fn main() {
    let mut window: PistonWindow = WindowSettings::new("training...", [450, 300])
        .samples(4)
        .build()
        .unwrap();

    let model = Polinomial::<6>::new();
    let mut trainer = Trainer::new(model);

    let mut epoch = 10;

    while let Some(_) = draw_piston_window(&mut window, |b| {
        for _ in 0..1000 {
            let done = trainer.train_step(&dataset_service(epoch));
            if !done {
                println!("local minimum for {epoch}");
                epoch += 1;
                epoch %= SPEED * 5;
                epoch = epoch.max(SPEED);
                break;
            }
        }

        let params = trainer.get_model_params();
        println!("{:?}", params);

        let root = b.into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(5)
            .build_cartesian_2d(-5f32..5f32, -60f32..140f32)?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                (-100..=100)
                    .map(|x| x as f32 / 20.0)
                    .map(|x| (x, polinomial(x))),
                &RED,
            ))?
            .label("y = x⁵ -4x⁴ -10x³ +40x² +9x -11")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .draw_series(LineSeries::new(
                (-100..=100)
                    .map(|x| x as f32 / 20.0)
                    .map(|x| (x, trainer.eval([x])[0])),
                &BLUE,
            ))?
            // .label(format!(
            //     "y = + {:.2}x⁵ + {:.2}x⁴ + {:.2}x³ + {:.2}x² + {:.2}x + {:.2}",
               
            //     params[5],
            //     params[4],
            //     params[3],
            //     params[2],
            //     params[1],
            //     params[0]
            // ))
            // .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE))
            ;

        let abs_max = 1 + epoch;

        chart.draw_series(LineSeries::new(
            (-abs_max..abs_max)
                .map(|x| x as f32 / SPEED as f32)
                .map(|x| (x, polinomial(x))),
            &GREEN_A700,
        ))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }) {}
}

fn dataset_service<const P: usize>(epoch: isize) -> Vec<DataPoint<P, 1, 1>> {
    let abs_max = 1 + epoch;
    (-abs_max..abs_max)
        .map(|x| x as f32 / SPEED as f32)
        // .map(|x| x as f32 / 10.)
        .map(|x| DataPoint {
            input: [x],
            output: [polinomial(x)],
        })
        .collect::<Vec<_>>()
}