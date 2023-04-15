use rand::Rng;
use std::f64;

// Activation functions
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

// Neural network structure
pub struct NeuralNetwork {
    weights: [Vec<Vec<f64>>; 3],
}

impl NeuralNetwork {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let weights = [
            (0..8)
                .map(|_| (0..3).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect(),
            (0..8)
                .map(|_| (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect(),
            (0..2)
                .map(|_| (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect(),
        ];
        NeuralNetwork { weights }
    }

    pub fn forward_pass(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut a = input.clone();
        for layer in &self.weights {
            a = layer
                .iter()
                .map(|weights| sigmoid(weights.iter().zip(&a).map(|(w, a)| w * a).sum()))
                .collect();
        }
        a
    }

    pub fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>, learning_rate: f64) {
        let mut a = vec![input.clone()];
        let mut z = vec![];

        // Forward pass
        for layer in &self.weights {
            let new_z: Vec<f64> = layer
                .iter()
                .map(|weights: &Vec<f64>| {
                    weights
                        .iter()
                        .zip(a.last().unwrap())
                        .map(|(w, a)| w * a)
                        .sum::<f64>()
                })
                .collect::<Vec<f64>>();
            z.push(new_z.clone());
            a.push(new_z.iter().map(|&x| sigmoid(x)).collect::<Vec<f64>>());
        }

        // Backpropagation
        let mut delta: Vec<f64> = a
            .last()
            .unwrap()
            .iter()
            .zip(target)
            .map(|(a, t)| (a - t) * sigmoid_derivative(*a))
            .collect();

        for (l, layer) in self.weights.iter_mut().enumerate().rev() {
            let new_delta: Vec<f64> = layer
                .iter()
                .map(|weights| weights.iter().zip(&delta).map(|(w, d)| w * d).sum())
                .collect();
            for (neuron_weights, d) in layer.iter_mut().zip(&delta) {
                for (w, a) in neuron_weights.iter_mut().zip(&a[l]) {
                    *w -= learning_rate * d * a;
                }
            }
            delta = new_delta
                .iter()
                .zip(&a[l])
                .map(|(d, a)| d * sigmoid_derivative(*a))
                .collect();
        }
    }

    pub fn train_iterations(&mut self, iterations: u32, learning_rate: f64) {
        for _ in 0..iterations {
            let input = vec![
                rand::thread_rng().gen_range(0..10) as f64,
                rand::thread_rng().gen_range(0..10) as f64,
                rand::thread_rng().gen_range(0..2) as f64,
            ];
            let target = match input[2] as usize {
                0 => vec![input[0] + input[1], 1.0],
                1 => {
                    if input[1] == 0.0 {
                        vec![0.0, 0.0]
                    } else {
                        vec![input[0] / input[1], 1.0]
                    }
                }
                _ => unreachable!(),
            };
            self.train(&input, &target, learning_rate);
        }
    }
}

// fn main() {
//     let mut network = NeuralNetwork::new();
//     network.train_iterations(10000, 0.1);

//     let test_input = vec![6.0, 3.0, 1.0];
//     let output = network.forward_pass(&test_input);
//     println!("Test input: {:?}", test_input);
//     println!("Output: {:?}", output);
// }
