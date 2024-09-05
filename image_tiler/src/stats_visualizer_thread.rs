use std::sync::mpsc::Receiver;



pub fn stats_thread(stats_rx: Receiver<Option<f32>>) {

    loop{
        while let Ok(d) = stats_rx.try_recv() {
            println!("cost: {d:?}");
        };
    }

 
}
