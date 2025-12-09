"""Simple neural network implementation for learning."""
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x)). Maps any value to (0, 1)."""
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Binary cross-entropy loss.

    Args:
        predicted: Predicted probabilities in (0, 1)
        target: True labels (0 or 1)

    Returns:
        Scalar loss value
    """
    # Clip predictions to avoid log(0)
    eps = 1e-7
    p = np.clip(predicted, eps, 1 - eps)

    # Binary cross-entropy: -[y * log(p) + (1-y) * log(1-p)]
    loss = -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))
    return float(loss)


class Network:
    """A fully-connected neural network."""

    def __init__(self, layer_sizes: list[int]):
        """
        Create a network with the given layer sizes.

        Args:
            layer_sizes: List of node counts per layer, e.g. [2, 3, 1]
                        for 2 inputs, 3 hidden, 1 output.
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Create weight matrices and bias vectors for each layer transition
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            # Weight matrix: (input_size, output_size)
            w = np.random.randn(input_size, output_size) * 0.1
            self.weights.append(w)

            # Bias vector: (output_size,)
            b = np.zeros(output_size)
            self.biases.append(b)

    def forward(self, x: np.ndarray, return_hidden: bool = False):
        """
        Compute the network output for a given input.

        Args:
            x: Input array of shape (input_size,)
            return_hidden: If True, also return activations from each layer

        Returns:
            output: Network output of shape (output_size,)
            hidden_activations: (only if return_hidden=True) List of activations
        """
        # Store values needed for backprop
        self._activations = [x]  # Input is first "activation"
        self._pre_activations = []  # z values before activation

        activation = x

        # Process each layer
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation: activation @ weights + bias
            z = activation @ w + b
            self._pre_activations.append(z)

            # Apply ReLU to all layers except the last (output layer is linear)
            if i < len(self.weights) - 1:
                activation = self._relu(z)
            else:
                activation = z  # Linear output for now

            self._activations.append(activation)

        if return_hidden:
            return activation, self._activations[1:]  # Exclude input
        return activation

    def backward(self, x: np.ndarray, target: np.ndarray):
        """
        Compute gradients for all weights and biases via backpropagation.

        Must call forward() first to populate intermediate values.

        Args:
            x: Input that was used in forward pass
            target: Target output (0 or 1 for binary classification)
        """
        # Initialize gradient lists
        self.weight_grads = [np.zeros_like(w) for w in self.weights]
        self.bias_grads = [np.zeros_like(b) for b in self.biases]

        # Get the output (after sigmoid)
        output = sigmoid(self._activations[-1])

        # Gradient of BCE loss w.r.t. sigmoid output: ∂L/∂p = -y/p + (1-y)/(1-p)
        eps = 1e-7
        p = np.clip(output, eps, 1 - eps)
        d_loss_d_p = -target / p + (1 - target) / (1 - p)

        # Gradient of sigmoid: ∂p/∂z = p * (1 - p)
        d_p_d_z = output * (1 - output)

        # Chain rule: ∂L/∂z_output = ∂L/∂p * ∂p/∂z
        delta = d_loss_d_p * d_p_d_z

        # Backpropagate through layers (from output to input)
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights: ∂L/∂W = activation^T @ delta
            # activation[i] is the input to layer i
            activation_in = self._activations[i]

            # Weight gradient: outer product of input activation and delta
            self.weight_grads[i] = np.outer(activation_in, delta)

            # Bias gradient: same as delta
            self.bias_grads[i] = delta.copy()

            # Propagate delta to previous layer (if not at input layer)
            if i > 0:
                # ∂L/∂activation = delta @ W^T
                delta = delta @ self.weights[i].T

                # Apply ReLU gradient: 0 if pre-activation was <= 0, else 1
                relu_grad = (self._pre_activations[i - 1] > 0).astype(float)
                delta = delta * relu_grad

    def step(self, learning_rate: float):
        """
        Update weights and biases using computed gradients.

        Must call backward() first.

        Args:
            learning_rate: Step size for gradient descent
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.weight_grads[i]
            self.biases[i] -= learning_rate * self.bias_grads[i]

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
