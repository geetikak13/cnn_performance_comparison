# src/utils.py
import numpy as np

def initialize_weights_he(dims: tuple, fan_in: int) -> np.ndarray:
    """
    Initializes weights using He initialization (for ReLU).

    Args:
        dims (tuple): Shape of the weight tensor.
        fan_in (int): The number of input units in the weight tensor.

    Returns:
        np.ndarray: Initialized weight tensor.
    """
    stddev = np.sqrt(2.0 / fan_in)
    # Use a specific seed for reproducibility if needed, but RandomState is better
    # return rng.randn(*dims).astype(np.float32) * stddev
    # Using default numpy random for simplicity here
    return np.random.randn(*dims).astype(np.float32) * stddev

def get_cnn_output_dims(h_in, w_in, fsize, stride=1, padding=0, pool_size=2, pool_stride=2):
    """
    Calculates the output dimensions after Conv2D and MaxPool2D layers.

    Args:
        h_in (int): Input height.
        w_in (int): Input width.
        fsize (int): Convolution filter size (assuming square).
        stride (int): Convolution stride.
        padding (int): Convolution padding.
        pool_size (int): Pooling kernel size (assuming square).
        pool_stride (int): Pooling stride.

    Returns:
        tuple: (h_conv_out, w_conv_out, h_pool_out, w_pool_out)
    """
    # Conv output size: floor(((W - K + 2P) / S) + 1)
    h_conv_out = int(np.floor(((h_in - fsize + 2 * padding) / stride) + 1))
    w_conv_out = int(np.floor(((w_in - fsize + 2 * padding) / stride) + 1))

    # Pool output size: floor(((W - K + 2P) / S) + 1) - Assuming padding=0 for pool
    h_pool_out = int(np.floor(((h_conv_out - pool_size) / pool_stride) + 1))
    w_pool_out = int(np.floor(((w_conv_out - pool_size) / pool_stride) + 1))

    return h_conv_out, w_conv_out, h_pool_out, w_pool_out


def calculate_dims(h0=28, w0=28, fsize=3, n_filters1=8, n_filters2=16, pool_size=2, pool_stride=2, hidden=128):
    """
    Calculates intermediate dimensions and flattened size for the specific CNN architecture.

    Args:
        h0 (int): Initial image height.
        w0 (int): Initial image width.
        fsize (int): Convolution filter size.
        n_filters1 (int): Number of filters in the first conv layer.
        n_filters2 (int): Number of filters in the second conv layer.
        pool_size (int): Max pooling kernel size.
        pool_stride (int): Max pooling stride.
        hidden (int): Number of units in the hidden dense layer.

    Returns:
        dict: Dictionary containing calculated dimensions:
              'c1_h', 'c1_w', 'p1_h', 'p1_w',
              'c2_h', 'c2_w', 'p2_h', 'p2_w',
              'flat_size', 'h0', 'w0'
    """
    # Layer 1: Conv1 -> Pool1
    c1_h, c1_w, p1_h, p1_w = get_cnn_output_dims(h0, w0, fsize, pool_size=pool_size, pool_stride=pool_stride)

    # Layer 2: Conv2 -> Pool2
    c2_h, c2_w, p2_h, p2_w = get_cnn_output_dims(p1_h, p1_w, fsize, pool_size=pool_size, pool_stride=pool_stride)

    # Flattened size after second pooling layer
    flat_size = n_filters2 * p2_h * p2_w

    dims = {
        'c1_h': c1_h, 'c1_w': c1_w, 'p1_h': p1_h, 'p1_w': p1_w,
        'c2_h': c2_h, 'c2_w': c2_w, 'p2_h': p2_h, 'p2_w': p2_w,
        'flat_size': flat_size,
        'h0': h0, # Include input dims for reference
        'w0': w0
    }
    print(f"Calculated Dimensions: {dims}")
    return dims


def initialize_cnn_weights_numpy(params, dims):
    """
    Initializes all CNN weights and biases as NumPy arrays based on params and dims.
    Also initializes corresponding velocity buffers for momentum.

    Args:
        params (dict): Dictionary containing hyperparameters like n_filters1, n_filters2, hidden, fsize.
        dims (dict): Dictionary containing calculated dimensions like flat_size.

    Returns:
        tuple: (weights, velocities)
               Two dictionaries containing the initialized weights/biases and zeroed velocity buffers.
    """
    n_filters1 = params['n_filters1']
    n_filters2 = params['n_filters2']
    fsize = params['fsize']
    hidden = params['hidden']
    flat_size = dims['flat_size']

    # Use a RandomState for reproducible weight initialization across runs if desired
    rng = np.random.RandomState(123) # Seed for reproducibility

    # Conv Layer 1: Input channels = 1 (grayscale MNIST)
    # Weight shape: (out_channels, in_channels, filter_height, filter_width)
    Wc1 = rng.randn(n_filters1, 1, fsize, fsize).astype(np.float32) * 0.1 # Example uses 0.1 scaling
    bc1 = np.zeros(n_filters1, np.float32)

    # Conv Layer 2: Input channels = n_filters1
    Wc2 = rng.randn(n_filters2, n_filters1, fsize, fsize).astype(np.float32) * 0.1
    bc2 = np.zeros(n_filters2, np.float32)

    # Dense Layer 1 (Hidden): Input size = flat_size
    # Weight shape: (input_features, output_features)
    W1 = initialize_weights_he((flat_size, hidden), fan_in=flat_size)
    b1 = np.zeros(hidden, np.float32)

    # Dense Layer 2 (Output): Input size = hidden
    W2 = initialize_weights_he((hidden, 10), fan_in=hidden) # Output size 10 for MNIST
    b2 = np.zeros(10, np.float32)

    weights = {'Wc1': Wc1, 'bc1': bc1, 'Wc2': Wc2, 'bc2': bc2, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # Initialize momentum velocity buffers (filled with zeros)
    velocities = {f'v{k}': np.zeros_like(v) for k, v in weights.items()}

    print("Initialized NumPy weights and velocity buffers.")
    return weights, velocities


if __name__ == '__main__':
    print("--- Testing Dimension Calculation ---")
    test_dims = calculate_dims()
    # Example assertion based on default params (28x28 -> 26x26 -> 13x13 -> 11x11 -> 5x5)
    # Flat size = n_filters2 * 5 * 5 = 16 * 25 = 400
    assert test_dims['flat_size'] == 400

    print("\n--- Testing Weight Initialization ---")
    test_params = {'n_filters1': 8, 'n_filters2': 16, 'fsize': 3, 'hidden': 128}
    weights_np, velocities_np = initialize_cnn_weights_numpy(test_params, test_dims)
    print("Weight shapes:")
    for name, w in weights_np.items():
        print(f"{name}: {w.shape}")
    print("\nVelocity shapes:")
    for name, v in velocities_np.items():
        print(f"{name}: {v.shape}")

    # Example check for weight shapes
    assert weights_np['Wc1'].shape == (test_params['n_filters1'], 1, test_params['fsize'], test_params['fsize'])
    assert weights_np['W1'].shape == (test_dims['flat_size'], test_params['hidden'])
    assert weights_np['W2'].shape == (test_params['hidden'], 10)
