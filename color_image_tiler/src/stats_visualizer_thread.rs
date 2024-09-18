use std::sync::mpsc::Receiver;



pub fn stats_thread(stats_rx: Receiver<(Option<f32>, usize)>) {
    loop{
        while let Ok(_) = stats_rx.recv() {
            // println!("cost: {d:?}");
        };
    }
}
