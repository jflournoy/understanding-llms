"""Visualization of neural network training."""
import numpy as np
import matplotlib.pyplot as plt
from src.network import Network, sigmoid, binary_cross_entropy


def train_and_record(net, X, y, epochs=500, learning_rate=0.5):
    """
    Train network and record weights, biases, and losses at each step.

    Returns:
        history: dict with 'weights', 'biases', 'losses', 'paths' lists
    """
    history = {
        'weights': [],
        'biases': [],
        'losses': [],
        'paths': [],  # Product of weights along each input->output path
    }

    for epoch in range(epochs):
        epoch_loss = 0

        # Train on each example
        for x, target in zip(X, y):
            output = sigmoid(net.forward(x))
            loss = binary_cross_entropy(output, target)
            epoch_loss += loss

            net.backward(x, target)
            net.step(learning_rate)

        # Record state after each epoch
        history['weights'].append([w.copy() for w in net.weights])
        history['biases'].append([b.copy() for b in net.biases])
        history['losses'].append(epoch_loss / len(X))
        history['paths'].append(compute_paths(net))

    return history


def compute_paths(net):
    """
    Compute the strength of each input->output path.

    For a [2, 2, 1] network, there are 4 paths:
    - input0 -> hidden0 -> output
    - input0 -> hidden1 -> output
    - input1 -> hidden0 -> output
    - input1 -> hidden1 -> output

    Path strength = product of weights along the path.
    """
    paths = []

    # For [2, hidden, 1] network
    n_inputs = net.weights[0].shape[0]
    n_hidden = net.weights[0].shape[1]

    for i in range(n_inputs):
        for h in range(n_hidden):
            # Weight from input i to hidden h
            w1 = net.weights[0][i, h]
            # Weight from hidden h to output
            w2 = net.weights[1][h, 0]
            paths.append({
                'input': i,
                'hidden': h,
                'strength': w1 * w2,
                'w1': w1,
                'w2': w2
            })

    return paths


def plot_training(history, save_path=None):
    """Create visualization of training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(len(history['losses']))

    # Plot 1: Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, history['losses'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)

    # Plot 2: Weight evolution (layer 0 - input to hidden)
    ax = axes[0, 1]
    w0_history = np.array([h[0] for h in history['weights']])  # shape: (epochs, n_in, n_hidden)
    n_in, n_hidden = w0_history.shape[1], w0_history.shape[2]
    for i in range(n_in):
        for j in range(n_hidden):
            ax.plot(epochs, w0_history[:, i, j], label=f'w[{i},{j}]')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Value')
    ax.set_title('Input→Hidden Weights')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Weight evolution (layer 1 - hidden to output)
    ax = axes[1, 0]
    w1_history = np.array([h[1] for h in history['weights']])  # shape: (epochs, n_hidden, n_out)
    n_hidden, n_out = w1_history.shape[1], w1_history.shape[2]
    for i in range(n_hidden):
        for j in range(n_out):
            ax.plot(epochs, w1_history[:, i, j], label=f'w[{i},{j}]')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Value')
    ax.set_title('Hidden→Output Weights')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Path strengths over time
    ax = axes[1, 1]
    n_paths = len(history['paths'][0])
    for p in range(n_paths):
        path_strengths = [history['paths'][e][p]['strength'] for e in epochs]
        i, h = history['paths'][0][p]['input'], history['paths'][0][p]['hidden']
        ax.plot(epochs, path_strengths, label=f'in{i}→h{h}→out')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Path Strength (w1 × w2)')
    ax.set_title('Input→Output Path Strengths')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def draw_network(net, x_input, target, title="Network State", ax=None, show_gradients=True):
    """
    Draw the network as a graph showing nodes, weights, and gradients.

    Args:
        net: The Network object
        x_input: Input to run forward pass with
        target: Target for computing gradients
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        show_gradients: Whether to show gradient values
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Run forward and backward to get current state
    output = sigmoid(net.forward(x_input))
    if show_gradients:
        net.backward(x_input, target)

    # Network structure
    layer_sizes = net.layer_sizes
    n_layers = len(layer_sizes)

    # Node positions
    max_nodes = max(layer_sizes)
    positions = {}  # (layer, node) -> (x, y)

    for layer_idx, n_nodes in enumerate(layer_sizes):
        x = layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
        # Center nodes vertically
        for node_idx in range(n_nodes):
            y = (node_idx + 0.5) / max_nodes + (max_nodes - n_nodes) / (2 * max_nodes)
            positions[(layer_idx, node_idx)] = (x, y)

    # Draw edges (weights) first so they're behind nodes
    for layer_idx in range(len(net.weights)):
        w = net.weights[layer_idx]
        grad = net.weight_grads[layer_idx] if show_gradients and hasattr(net, 'weight_grads') else None

        n_in, n_out = w.shape
        for i in range(n_in):
            for j in range(n_out):
                x1, y1 = positions[(layer_idx, i)]
                x2, y2 = positions[(layer_idx + 1, j)]

                weight = w[i, j]

                # Color by sign, thickness by magnitude
                color = 'blue' if weight > 0 else 'red'
                alpha = min(0.9, 0.2 + abs(weight) / 5)
                linewidth = 0.5 + abs(weight) * 0.5

                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                       linewidth=linewidth, zorder=1)

                # Label with weight value
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                # Offset labels to avoid overlap
                offset = (j - n_out/2 + 0.5) * 0.02

                label = f"{weight:.2f}"
                if show_gradients and grad is not None:
                    g = grad[i, j]
                    label += f"\n(∇{g:.2f})"

                ax.annotate(label, (mid_x, mid_y + offset), fontsize=7,
                           ha='center', va='center', color='darkgray')

    # Draw nodes
    for (layer_idx, node_idx), (x, y) in positions.items():
        # Determine node value/activation
        if layer_idx == 0:
            # Input layer
            value = x_input[node_idx]
            label = f"in{node_idx}\n{value:.1f}"
            color = 'lightgreen'
        elif layer_idx == n_layers - 1:
            # Output layer
            value = output[node_idx] if len(output.shape) > 0 else output
            label = f"out\n{float(value):.3f}"
            color = 'lightcoral'
        else:
            # Hidden layer
            activation = net._activations[layer_idx][node_idx]
            bias = net.biases[layer_idx - 1][node_idx]
            label = f"h{node_idx}\na={activation:.2f}\nb={bias:.2f}"
            if show_gradients and hasattr(net, 'bias_grads'):
                bg = net.bias_grads[layer_idx - 1][node_idx]
                label += f"\n∇b={bg:.2f}"
            color = 'lightyellow' if activation > 0 else 'lightgray'

        circle = plt.Circle((x, y), 0.06, color=color, ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.annotate(label, (x, y), fontsize=8, ha='center', va='center', zorder=3)

    # Add output bias
    if len(net.biases) > 0:
        out_bias = net.biases[-1][0]
        out_x, out_y = positions[(n_layers - 1, 0)]
        bias_label = f"bias={out_bias:.2f}"
        if show_gradients and hasattr(net, 'bias_grads'):
            bias_label += f"\n∇={net.bias_grads[-1][0]:.2f}"
        ax.annotate(bias_label, (out_x + 0.08, out_y), fontsize=8, ha='left', va='center')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Legend
    ax.plot([], [], 'b-', linewidth=2, label='Positive weight')
    ax.plot([], [], 'r-', linewidth=2, label='Negative weight')
    ax.legend(loc='upper left', fontsize=8)

    return ax


def draw_training_steps(net, X, y, steps=[0, 1, 5, 20, 100, 500],
                        learning_rate=0.5, save_path=None):
    """
    Draw network state at various training steps for a single input.

    Shows how weights and gradients evolve during training.
    """
    # Reset network with same initialization
    np.random.seed(123)
    net = Network(net.layer_sizes)
    for i in range(len(net.weights)):
        fan_in = net.weights[i].shape[0]
        net.weights[i] = np.random.randn(*net.weights[i].shape) * np.sqrt(2.0 / fan_in)

    # Use first input/target for visualization
    x_input = X[0]
    target = y[0]

    n_plots = len(steps)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    step = 0
    plot_idx = 0

    while plot_idx < len(steps) and step <= max(steps):
        if step in steps:
            # Compute loss for this state
            output = sigmoid(net.forward(x_input))
            loss = binary_cross_entropy(output, target)

            title = f"Step {step} | Loss: {loss:.4f} | Input: {x_input} → Target: {target[0]}"
            draw_network(net, x_input, target, title=title, ax=axes[plot_idx])
            plot_idx += 1

        # Train one step on all examples
        for x, t in zip(X, y):
            sigmoid(net.forward(x))
            net.backward(x, t)
            net.step(learning_rate)
        step += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_biases(history, save_path=None):
    """Plot bias evolution over training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(len(history['losses']))

    # Hidden layer biases
    ax = axes[0]
    b0_history = np.array([h[0] for h in history['biases']])
    for i in range(b0_history.shape[1]):
        ax.plot(epochs, b0_history[:, i], label=f'bias[{i}]')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Bias Value')
    ax.set_title('Hidden Layer Biases')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Output layer biases
    ax = axes[1]
    b1_history = np.array([h[1] for h in history['biases']])
    for i in range(b1_history.shape[1]):
        ax.plot(epochs, b1_history[:, i], label=f'bias[{i}]')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Bias Value')
    ax.set_title('Output Layer Biases')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def train_xor(show_plots=False, save_plots=True):
    """Train on XOR and visualize."""
    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ], dtype=float)

    # Create network with better initialization
    np.random.seed(123)  # Different seed
    net = Network([2, 4, 1])  # 2 inputs, 4 hidden, 1 output

    # Use larger initial weights (Xavier-like initialization)
    for i in range(len(net.weights)):
        fan_in = net.weights[i].shape[0]
        net.weights[i] = np.random.randn(*net.weights[i].shape) * np.sqrt(2.0 / fan_in)

    # Train and record
    print("Training on XOR...")
    history = train_and_record(net, X, y, epochs=2000, learning_rate=0.5)

    # Check final predictions
    print("\nFinal predictions:")
    for x, target in zip(X, y):
        pred = sigmoid(net.forward(x))
        print(f"  {x} -> {pred[0]:.3f} (target: {target[0]})")

    print(f"\nFinal loss: {history['losses'][-1]:.4f}")

    # Plot
    if save_plots:
        plot_training(history, save_path="xor_training.png")
        plot_biases(history, save_path="xor_biases.png")
    elif show_plots:
        plot_training(history)
        plot_biases(history)

    return net, history


def draw_all_inputs_at_step(net, X, y, step, learning_rate=0.5, save_path=None):
    """
    Train network to a specific step, then show how it handles all 4 XOR inputs.
    """
    # Reset network
    np.random.seed(123)
    net = Network(net.layer_sizes)
    for i in range(len(net.weights)):
        fan_in = net.weights[i].shape[0]
        net.weights[i] = np.random.randn(*net.weights[i].shape) * np.sqrt(2.0 / fan_in)

    # Train to the specified step
    for _ in range(step):
        for x, t in zip(X, y):
            sigmoid(net.forward(x))
            net.backward(x, t)
            net.step(learning_rate)

    # Now draw network state for each input
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (x_input, target) in enumerate(zip(X, y)):
        output = sigmoid(net.forward(x_input))
        loss = binary_cross_entropy(output, target)
        title = f"Input: {x_input} → Target: {target[0]:.0f} | Pred: {float(output[0]):.3f} | Loss: {loss:.4f}"
        draw_network(net, x_input, target, title=title, ax=axes[idx])

    plt.suptitle(f"XOR Network After {step} Training Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # Use 4 hidden nodes - XOR needs at least 2 working hidden nodes
    net = Network([2, 4, 1])

    # Show network after training converges
    draw_all_inputs_at_step(net, X, y, step=2000, save_path="xor_trained.png")
