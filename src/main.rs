
mod perceptron;
use perceptron::Perceptron;

fn main() {
    let perceptron = Perceptron::new(2).add_layer(2).add_layer(2).add_layer(1);

    


}
