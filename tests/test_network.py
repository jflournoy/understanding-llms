"""Tests for neural network construction and operations."""
import numpy as np
import pytest


class TestNetworkConstruction:
    """Tests for building a network with the right structure."""

    def test_network_has_correct_number_of_weight_matrices(self):
        """A [2, 3, 1] network should have 2 weight matrices (one per layer transition)."""
        from src.network import Network

        net = Network([2, 3, 1])

        assert len(net.weights) == 2

    def test_weight_matrix_shapes(self):
        """Weight matrices should have shape (input_size, output_size) for each layer."""
        from src.network import Network

        net = Network([2, 3, 1])

        # First layer: 2 inputs -> 3 hidden nodes
        assert net.weights[0].shape == (2, 3)
        # Second layer: 3 hidden -> 1 output
        assert net.weights[1].shape == (3, 1)

    def test_bias_vector_shapes(self):
        """Bias vectors should have shape (layer_size,) for each non-input layer."""
        from src.network import Network

        net = Network([2, 3, 1])

        assert len(net.biases) == 2
        assert net.biases[0].shape == (3,)  # hidden layer
        assert net.biases[1].shape == (1,)  # output layer


class TestForwardPass:
    """Tests for computing network output from input."""

    def test_output_shape_matches_final_layer(self):
        """Forward pass should return array matching output layer size."""
        from src.network import Network

        net = Network([2, 3, 1])
        output = net.forward(np.array([1.0, 2.0]))

        assert output.shape == (1,)

    def test_output_shape_with_multiple_outputs(self):
        """Network with multiple output nodes should return correct shape."""
        from src.network import Network

        net = Network([2, 3, 4])
        output = net.forward(np.array([1.0, 2.0]))

        assert output.shape == (4,)

    def test_hidden_activations_are_nonnegative(self):
        """ReLU should ensure all hidden layer activations are >= 0."""
        from src.network import Network

        net = Network([2, 3, 1])
        # Run forward and check hidden layer activations
        output, hidden_activations = net.forward(np.array([1.0, -1.0]), return_hidden=True)

        # Only check hidden layers (all except the last, which is output)
        for activation in hidden_activations[:-1]:
            assert np.all(activation >= 0), "ReLU should make all activations non-negative"

    def test_known_weights_produce_expected_output(self):
        """With specific weights, we should get a predictable output."""
        from src.network import Network

        net = Network([2, 2, 1])
        # Set known weights and biases
        net.weights[0] = np.array([[1.0, 0.0],   # input 0 -> hidden
                                    [0.0, 1.0]])  # input 1 -> hidden
        net.biases[0] = np.array([0.0, 0.0])

        net.weights[1] = np.array([[1.0],   # hidden 0 -> output
                                    [1.0]])  # hidden 1 -> output
        net.biases[1] = np.array([0.0])

        # Input [3, 4] with identity-like weights:
        # hidden = ReLU([3, 4]) = [3, 4]
        # output = [3 + 4] = [7]
        output = net.forward(np.array([3.0, 4.0]))

        assert np.allclose(output, [7.0])

    def test_relu_clips_negative_values(self):
        """ReLU should clip negative pre-activations to zero."""
        from src.network import Network

        net = Network([2, 2, 1])
        # Weights that produce negative pre-activations
        net.weights[0] = np.array([[1.0, -1.0],
                                    [1.0, -1.0]])
        net.biases[0] = np.array([0.0, 0.0])
        net.weights[1] = np.array([[1.0], [1.0]])
        net.biases[1] = np.array([0.0])

        # Input [1, 2]:
        # pre-activation = [1+2, -1-2] = [3, -3]
        # after ReLU = [3, 0]
        # output = 3 + 0 = 3
        output = net.forward(np.array([1.0, 2.0]))

        assert np.allclose(output, [3.0])

    def test_zero_input(self):
        """Network should handle zero input gracefully."""
        from src.network import Network

        net = Network([2, 3, 1])
        output = net.forward(np.array([0.0, 0.0]))

        # With zero input, output depends only on biases
        # Should not crash and should return valid number
        assert output.shape == (1,)
        assert np.isfinite(output[0])


class TestSigmoid:
    """Tests for the sigmoid activation function."""

    def test_sigmoid_output_range(self):
        """Sigmoid should always output values in [0, 1]."""
        from src.network import sigmoid

        # Test various inputs (not too extreme to avoid float precision issues)
        inputs = np.array([-10, -1, 0, 1, 10])
        outputs = sigmoid(inputs)

        assert np.all(outputs > 0)
        assert np.all(outputs < 1)

        # For extreme values, sigmoid saturates to 0 or 1 due to float precision
        # This is expected behavior
        assert sigmoid(-100) >= 0
        assert sigmoid(100) <= 1

    def test_sigmoid_of_zero_is_half(self):
        """Sigmoid(0) should equal 0.5."""
        from src.network import sigmoid

        assert np.isclose(sigmoid(0), 0.5)

    def test_sigmoid_symmetry(self):
        """Sigmoid(-x) should equal 1 - sigmoid(x)."""
        from src.network import sigmoid

        x = np.array([1, 2, 3])
        assert np.allclose(sigmoid(-x), 1 - sigmoid(x))


class TestBinaryCrossEntropyLoss:
    """Tests for binary cross-entropy loss function."""

    def test_loss_returns_scalar(self):
        """Loss should return a single scalar value."""
        from src.network import binary_cross_entropy

        loss = binary_cross_entropy(predicted=np.array([0.5]), target=np.array([1.0]))

        assert np.isscalar(loss) or loss.shape == ()

    def test_loss_is_nonnegative(self):
        """Loss should always be >= 0."""
        from src.network import binary_cross_entropy

        # Various predictions and targets
        test_cases = [
            (np.array([0.1]), np.array([0.0])),
            (np.array([0.9]), np.array([1.0])),
            (np.array([0.5]), np.array([0.0])),
            (np.array([0.5]), np.array([1.0])),
        ]

        for pred, target in test_cases:
            loss = binary_cross_entropy(pred, target)
            assert loss >= 0, f"Loss should be >= 0, got {loss}"

    def test_perfect_prediction_has_low_loss(self):
        """Correct confident predictions should have near-zero loss."""
        from src.network import binary_cross_entropy

        # Predict 0.99 when target is 1
        loss = binary_cross_entropy(np.array([0.99]), np.array([1.0]))
        assert loss < 0.1

        # Predict 0.01 when target is 0
        loss = binary_cross_entropy(np.array([0.01]), np.array([0.0]))
        assert loss < 0.1

    def test_wrong_prediction_has_high_loss(self):
        """Wrong confident predictions should have high loss."""
        from src.network import binary_cross_entropy

        # Predict 0.01 when target is 1 (confidently wrong)
        loss = binary_cross_entropy(np.array([0.01]), np.array([1.0]))
        assert loss > 2.0  # -log(0.01) ≈ 4.6

    def test_known_loss_value(self):
        """Test against hand-calculated loss value."""
        from src.network import binary_cross_entropy

        # loss = -[1 * log(0.5) + 0 * log(0.5)] = -log(0.5) ≈ 0.693
        loss = binary_cross_entropy(np.array([0.5]), np.array([1.0]))
        assert np.isclose(loss, 0.693, atol=0.01)

    def test_handles_edge_case_prediction_zero(self):
        """Should not crash or return inf when prediction is near 0."""
        from src.network import binary_cross_entropy

        # This would be log(0) = -inf without clipping
        loss = binary_cross_entropy(np.array([0.0]), np.array([1.0]))

        assert np.isfinite(loss)

    def test_handles_edge_case_prediction_one(self):
        """Should not crash or return inf when prediction is near 1."""
        from src.network import binary_cross_entropy

        # This would be log(0) = -inf without clipping (from 1-p term)
        loss = binary_cross_entropy(np.array([1.0]), np.array([0.0]))

        assert np.isfinite(loss)


class TestBackpropagation:
    """Tests for computing gradients via backpropagation."""

    def test_backward_creates_weight_gradients(self):
        """After backward(), network should have weight gradients."""
        from src.network import Network, sigmoid, binary_cross_entropy

        net = Network([2, 3, 1])
        x = np.array([1.0, 2.0])
        target = np.array([1.0])

        # Forward pass
        output = sigmoid(net.forward(x))
        loss = binary_cross_entropy(output, target)

        # Backward pass
        net.backward(x, target)

        # Should now have gradients
        assert hasattr(net, 'weight_grads')
        assert len(net.weight_grads) == len(net.weights)

    def test_weight_gradients_have_correct_shapes(self):
        """Weight gradients should match weight shapes."""
        from src.network import Network, sigmoid, binary_cross_entropy

        net = Network([2, 3, 1])
        x = np.array([1.0, 2.0])
        target = np.array([1.0])

        output = sigmoid(net.forward(x))
        net.backward(x, target)

        for i, (w, grad) in enumerate(zip(net.weights, net.weight_grads)):
            assert grad.shape == w.shape, f"Gradient {i} shape {grad.shape} != weight shape {w.shape}"

    def test_bias_gradients_have_correct_shapes(self):
        """Bias gradients should match bias shapes."""
        from src.network import Network, sigmoid, binary_cross_entropy

        net = Network([2, 3, 1])
        x = np.array([1.0, 2.0])
        target = np.array([1.0])

        output = sigmoid(net.forward(x))
        net.backward(x, target)

        assert hasattr(net, 'bias_grads')
        for i, (b, grad) in enumerate(zip(net.biases, net.bias_grads)):
            assert grad.shape == b.shape, f"Bias gradient {i} shape {grad.shape} != bias shape {b.shape}"

    def test_step_changes_weights(self):
        """After step(), weights should be different."""
        from src.network import Network, sigmoid, binary_cross_entropy

        net = Network([2, 3, 1])
        x = np.array([1.0, 2.0])
        target = np.array([1.0])

        # Store original weights
        original_weights = [w.copy() for w in net.weights]

        # Forward, backward, step
        output = sigmoid(net.forward(x))
        net.backward(x, target)
        net.step(learning_rate=0.1)

        # Weights should have changed
        weights_changed = False
        for orig, new in zip(original_weights, net.weights):
            if not np.allclose(orig, new):
                weights_changed = True
                break

        assert weights_changed, "Weights should change after step()"

    def test_training_reduces_loss(self):
        """Multiple training steps should reduce loss."""
        from src.network import Network, sigmoid, binary_cross_entropy

        net = Network([2, 3, 1])
        x = np.array([1.0, 0.0])
        target = np.array([1.0])

        # Initial loss
        output = sigmoid(net.forward(x))
        initial_loss = binary_cross_entropy(output, target)

        # Train for several steps
        for _ in range(100):
            output = sigmoid(net.forward(x))
            net.backward(x, target)
            net.step(learning_rate=0.1)

        # Final loss
        output = sigmoid(net.forward(x))
        final_loss = binary_cross_entropy(output, target)

        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"

    def test_gradients_match_numerical_approximation(self):
        """Analytical gradients should match numerical gradients (gradient check)."""
        from src.network import Network, sigmoid, binary_cross_entropy

        net = Network([2, 2, 1])
        x = np.array([1.0, 2.0])
        target = np.array([1.0])

        # Compute analytical gradients
        output = sigmoid(net.forward(x))
        net.backward(x, target)

        # Check one weight numerically
        eps = 1e-5
        i, j = 0, 0  # First weight in first layer

        # Compute numerical gradient for weights[0][i,j]
        original = net.weights[0][i, j]

        net.weights[0][i, j] = original + eps
        loss_plus = binary_cross_entropy(sigmoid(net.forward(x)), target)

        net.weights[0][i, j] = original - eps
        loss_minus = binary_cross_entropy(sigmoid(net.forward(x)), target)

        net.weights[0][i, j] = original  # Restore

        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical_grad = net.weight_grads[0][i, j]

        assert np.isclose(numerical_grad, analytical_grad, rtol=1e-3), \
            f"Numerical: {numerical_grad}, Analytical: {analytical_grad}"
