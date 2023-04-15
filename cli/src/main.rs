// cli/src/main.rs

use std::io;
use neural_app::NeuralNetwork;
use rand::Rng;

fn main() {
    let mut network = NeuralNetwork::new();

    println!("Neural Network Calculator");

    loop {
        println!("\nMenu:");
        println!("1. Train the network");
        println!("2. Calculate an operation");
        println!("3. Exit");

        let mut choice = String::new();
        io::stdin()
            .read_line(&mut choice)
            .expect("Failed to read input");

        match choice.trim().parse::<u32>() {
            Ok(1) => train_network_menu(&mut network),
            Ok(2) => calculate_operation(&mut network),
            Ok(3) => break,
            _ => println!("Invalid choice, please try again"),
        }
    }
}

fn train_network(network: &mut NeuralNetwork, iterations: usize, learning_rate: f64) {
    let mut rng = rand::thread_rng();

    for _ in 0..iterations {
        // Generate random inputs
        let num1 = rng.gen_range(0.0..10.0);
        let num2 = rng.gen_range(0.0..10.0);
        let operator = rng.gen_range(0..4) as f64;

        let input_values = vec![num1, num2, operator];

        // Calculate the expected output
        let (expected_output_1, expected_output_2) = perform_operation(&input_values);
        let expected_output = vec![expected_output_1, expected_output_2];

        // Train the network with the input and expected output
        network.train(&input_values, &expected_output, learning_rate);
    }
}


fn train_network_menu(network: &mut NeuralNetwork) {
    println!("Enter the number of iterations:");

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    let iterations = input.trim().parse::<usize>().unwrap_or(0);
    if iterations > 0 {
        train_network(network, iterations, 0.1);
        println!("Training completed.");
    } else {
        println!("Invalid number of iterations.");
    }
}

fn perform_operation(input_values: &[f64]) -> (f64, f64) {
    let num1 = input_values[0];
    let num2 = input_values[1];
    let operator = input_values[2] as u32;

    let result = match operator {
        0 => num1 + num2,
        1 => num1 - num2,
        2 => num1 * num2,
        3 => {
            if num2 != 0.0 {
                num1 / num2
            } else {
                return (std::f64::MAX, 0.0);
            }
        }
        _ => std::f64::NAN,
    };

    (result, 1.0)
}



fn calculate_operation(network: &mut NeuralNetwork) {
    println!("Enter two single-digit numbers and an operator (0: add, 1: subtract, 2: multiply, 3: divide):");

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    let input_values: Vec<f64> = input
        .trim()
        .split_whitespace()
        .map(|s| s.parse().unwrap_or(0.0))
        .collect();

    if input_values.len() == 3 {
        let (correct_output_1, correct_output_2) = perform_operation(&input_values);
        let nn_output = network.forward_pass(&input_values);

        println!("Correct Output: [{:.2}, {:.2}]", correct_output_1, correct_output_2);
        println!("Neural Network Output: [{:.2}, {:.2}]", nn_output[0], nn_output[1]);
    } else {
        println!("Invalid input format.");
    }
}


