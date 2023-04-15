use neural_app::{NeuralNetwork, sigmoid, sigmoid_derivative};

// Add this to the top of your file to import additional macros and functionality for testing
#[cfg(test)]
use assert_approx_eq::assert_approx_eq;

// Test module
#[cfg(test)]
mod tests {

    use super::*;

    // Test the sigmoid function
    #[test]
    fn test_sigmoid() {
        assert_approx_eq!(sigmoid(0.0), 0.5, 1e-7);
        assert_approx_eq!(sigmoid(2.0), 0.880797, 1e-6);
        assert_approx_eq!(sigmoid(-2.0), 0.119203, 1e-6);
    }

    // Test the sigmoid_derivative function
    #[test]
    fn test_sigmoid_derivative() {
        assert_approx_eq!(sigmoid_derivative(0.0), 0.0, 1e-7);
        assert_approx_eq!(sigmoid_derivative(0.5), 0.25, 1e-7);
        assert_approx_eq!(sigmoid_derivative(0.25), 0.1875, 1e-7);
    }

    // Test the neural network's forward pass
    #[test]
    fn test_forward_pass() {
        let mut network = NeuralNetwork::new();

        let input = vec![2.0, 5.0, 0.0];
        let output = network.forward_pass(&input);

        assert_eq!(output.len(), 2);
    }

    // Test the neural network's training function
    #[test]
    fn test_train() {
        let mut network = NeuralNetwork::new();

        let input = vec![2.0, 5.0, 0.0];
        let target = vec![7.0, 1.0];
        network.train(&input, &target, 0.1);

        let output = network.forward_pass(&input);

        assert_eq!(output.len(), 2);
    }
}
