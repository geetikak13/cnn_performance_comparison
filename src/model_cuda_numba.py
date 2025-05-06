# src/model_cuda_numba.py
import numpy as np
from numba import cuda
import math
import time

# ==============================================================================
# CUDA Kernels (Adapted from works_numba.py)
# ==============================================================================
# These kernels run on the GPU. They use cuda.grid(ndim) to get thread indices
# and often use grid-stride loops to handle inputs larger than the grid size.

@cuda.jit(device=False, fastmath=True) # device=False makes it a callable kernel
def conv2d_relu_cuda(inp, W, b, out):
    """Performs 2D convolution followed by ReLU activation (CUDA Numba)."""
    B, C_in, H_in, W_in = inp.shape
    F_out, _, f_h, f_w = W.shape
    H_out = H_in - f_h + 1
    W_out = W_in - f_w + 1

    # --- Grid-Stride Loop Setup ---
    # Use 3D grid: (Batch, Filters, OutputPixels)
    start_bi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    start_fo = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    start_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    grid_stride_bi = cuda.gridDim.x * cuda.blockDim.x
    grid_stride_fo = cuda.gridDim.y * cuda.blockDim.y
    grid_stride_idx = cuda.gridDim.z * cuda.blockDim.z

    # --- Main Loop ---
    for bi in range(start_bi, B, grid_stride_bi):
        for fo in range(start_fo, F_out, grid_stride_fo):
            for idx in range(start_idx, H_out * W_out, grid_stride_idx):
                # Calculate output coordinates from linear index
                oy = idx // W_out
                ox = idx % W_out

                # Compute convolution sum
                acc = b[fo] # Start with bias
                for ci in range(C_in):
                    for u in range(f_h):
                        for v in range(f_w):
                            iy = oy + u
                            ix = ox + v
                            # Note: No explicit boundary checks needed if input size matches assumptions
                            acc += inp[bi, ci, iy, ix] * W[fo, ci, u, v]

                # Apply ReLU
                out[bi, fo, oy, ox] = acc if acc > 0.0 else 0.0


@cuda.jit(device=False, fastmath=True)
def maxpool_cuda(inp, out):
    """Performs 2x2 Max Pooling with stride 2 (CUDA Numba)."""
    B, C, H_in, W_in = inp.shape
    H_out = H_in // 2
    W_out = W_in // 2

    # --- Grid-Stride Loop Setup ---
    # Use 3D grid: (Batch, Channels, OutputPixels)
    start_bi = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    start_ci = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    start_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    grid_stride_bi = cuda.gridDim.x * cuda.blockDim.x
    grid_stride_ci = cuda.gridDim.y * cuda.blockDim.y
    grid_stride_idx = cuda.gridDim.z * cuda.blockDim.z

    # --- Main Loop ---
    for bi in range(start_bi, B, grid_stride_bi):
        for ci in range(start_ci, C, grid_stride_ci):
            for idx in range(start_idx, H_out * W_out, grid_stride_idx):
                # Calculate output coordinates
                oy = idx // W_out
                ox = idx % W_out

                # Input window coordinates (top-left)
                iy_start = oy * 2
                ix_start = ox * 2

                # Find max in 2x2 window
                # Initialize max_val carefully for floating point
                max_val = -np.inf # Or inp[bi, ci, iy_start, ix_start]
                v00 = inp[bi, ci, iy_start,     ix_start    ]
                v01 = inp[bi, ci, iy_start,     ix_start + 1]
                v10 = inp[bi, ci, iy_start + 1, ix_start    ]
                v11 = inp[bi, ci, iy_start + 1, ix_start + 1]
                if v00 > max_val: max_val = v00
                if v01 > max_val: max_val = v01
                if v10 > max_val: max_val = v10
                if v11 > max_val: max_val = v11
                out[bi, ci, oy, ox] = max_val


@cuda.jit(device=False, fastmath=True)
def dense_relu_cuda(inp, W, b, out):
    """Performs Dense (Fully Connected) layer followed by ReLU (CUDA Numba)."""
    B, D_in = inp.shape
    _, K_out = W.shape

    # --- Grid Setup (2D: Batch, OutputFeatures) ---
    i, j = cuda.grid(2) # i corresponds to batch, j to output feature

    # --- Boundary Check ---
    if i < B and j < K_out:
        # Compute dot product
        acc = b[j] # Start with bias
        for k in range(D_in):
            acc += inp[i, k] * W[k, j]
        # Apply ReLU
        out[i, j] = acc if acc > 0.0 else 0.0


@cuda.jit(device=False, fastmath=True)
def dense_softmax_cuda(inp, W, b, logits, probs):
    """Performs Dense layer followed by Softmax activation (CUDA Numba)."""
    B, D_in = inp.shape
    _, K_out = W.shape # K_out = NumClasses

    # --- Grid Setup (1D: Batch) ---
    # Each thread handles one sample in the batch
    i = cuda.grid(1)

    # --- Boundary Check ---
    if i < B:
        # --- Calculate Logits ---
        # Use shared memory for reduction (max, sum) for better performance,
        # but for simplicity, calculate row-wise like CPU version first.
        max_logit_in_row = -np.inf
        # Calculate and store logits for the current row (i)
        for j in range(K_out):
            acc = b[j]
            for k in range(D_in):
                acc += inp[i, k] * W[k, j]
            logits[i, j] = acc
            if acc > max_logit_in_row:
                max_logit_in_row = acc

        # --- Calculate Softmax Probabilities ---
        sum_exp = 0.0
        for j in range(K_out):
            # Subtract max logit for stability
            exp_val = math.exp(logits[i, j] - max_logit_in_row)
            probs[i, j] = exp_val # Store unnormalized
            sum_exp += exp_val

        # --- Normalize ---
        inv_sum_exp = 1.0 / sum_exp if sum_exp > 1e-9 else 0.0
        for j in range(K_out):
            probs[i, j] *= inv_sum_exp


@cuda.jit(device=False) # No fastmath needed
def backprop_output_cuda(probs, y, dlogits):
    """Calculates gradient of CrossEntropyLoss w.r.t. logits (CUDA Numba)."""
    B, K = probs.shape
    # --- Grid Setup (1D: Batch) ---
    i = cuda.grid(1)

    # --- Boundary Check ---
    if i < B:
        true_class_idx = y[i] # y is on GPU
        for j in range(K):
            target_prob = 1.0 if j == true_class_idx else 0.0
            dlogits[i, j] = probs[i, j] - target_prob


# Note: CUDA backpropagation for dense/conv layers is complex to write manually.
# The FC gradients are calculated on the CPU after copying
# activations and gradients back. We follow that pattern for simplicity,
# although a full GPU backprop (e.g., using cuBLAS for GEMM) would be faster.

# ==============================================================================
# CUDA Model Runner Class
# ==============================================================================

class CUDAModelRunner:
    """Manages GPU resources, kernel launches, and training/evaluation for CUDA."""

    def __init__(self, params, dims, initial_weights, initial_velocities):
        """
        Initializes GPU resources, copies weights, defines launch configs.

        Args:
            params (dict): Hyperparameters.
            dims (dict): Layer dimensions.
            initial_weights (dict): Initial weights (NumPy arrays from CPU).
            initial_velocities (dict): Initial velocities (NumPy arrays from CPU).
        """
        self.params = params
        self.dims = dims
        self.batch_size = params['batch_size']
        self.n_filters1 = params['n_filters1']
        self.n_filters2 = params['n_filters2']
        self.hidden = params['hidden']
        self.flat_size = dims['flat_size']
        self.lr = params['lr']
        self.momentum = params['momentum']

        # Keep CPU copies of weights and velocities for CPU-based updates
        self.weights_host = initial_weights
        self.velocities_host = initial_velocities

        print("Initializing CUDA resources...")
        # --- Copy weights/biases to GPU ---
        self.d_weights = {}
        try:
            for k, v in self.weights_host.items():
                self.d_weights[k] = cuda.to_device(v)
            print("Weights copied to GPU.")
        except Exception as e:
            print(f"Error copying weights to GPU: {e}")
            raise

        # --- Pre-allocate GPU activation/gradient buffers ---
        # These stay on the GPU during training epochs.
        # Activations:
        self.d_conv1_out = cuda.device_array((self.batch_size, self.n_filters1, dims['c1_h'], dims['c1_w']), np.float32)
        self.d_pool1_out = cuda.device_array((self.batch_size, self.n_filters1, dims['p1_h'], dims['p1_w']), np.float32)
        self.d_conv2_out = cuda.device_array((self.batch_size, self.n_filters2, dims['c2_h'], dims['c2_w']), np.float32)
        self.d_pool2_out = cuda.device_array((self.batch_size, self.n_filters2, dims['p2_h'], dims['p2_w']), np.float32)
        self.d_flat_out = cuda.device_array((self.batch_size, self.flat_size), np.float32) # Allocate explicitly
        self.d_hid_out = cuda.device_array((self.batch_size, self.hidden), np.float32)
        self.d_logits = cuda.device_array((self.batch_size, 10), np.float32)
        self.d_probs = cuda.device_array((self.batch_size, 10), np.float32)
        # Gradients (only output gradient needed on GPU for CPU backprop approach):
        self.d_dlogits = cuda.device_array((self.batch_size, 10), np.float32)
        # Input/Target buffers (reused per batch)
        # Use dimensions from dims dict if available
        h0_dim = dims.get('h0', 28) # Default to 28 if not found
        w0_dim = dims.get('w0', 28) # Default to 28 if not found
        self.d_xb = cuda.device_array((self.batch_size, 1, h0_dim, w0_dim), np.float32)
        self.d_yb = cuda.device_array((self.batch_size,), np.int32)
        print("GPU buffers allocated.")


        # --- Define Launch configurations ---
        # These need tuning based on GPU architecture and problem size.
        # For Conv/Pool (3D grid: B, F/C, H*W)
        threads_per_block_3d = (8, 8, 8) # 512 threads, adjust as needed
        # Calculate grid dimensions (number of blocks)
        def get_grid_dim(total_size, block_size):
            return (total_size + block_size - 1) // block_size

        self.tp_conv = threads_per_block_3d
        self.blocks_c1 = (get_grid_dim(self.batch_size, self.tp_conv[0]),
                          get_grid_dim(self.n_filters1, self.tp_conv[1]),
                          get_grid_dim(dims['c1_h'] * dims['c1_w'], self.tp_conv[2]))
        self.blocks_p1 = (get_grid_dim(self.batch_size, self.tp_conv[0]),
                          get_grid_dim(self.n_filters1, self.tp_conv[1]),
                          get_grid_dim(dims['p1_h'] * dims['p1_w'], self.tp_conv[2]))
        self.blocks_c2 = (get_grid_dim(self.batch_size, self.tp_conv[0]),
                          get_grid_dim(self.n_filters2, self.tp_conv[1]),
                          get_grid_dim(dims['c2_h'] * dims['c2_w'], self.tp_conv[2]))
        self.blocks_p2 = (get_grid_dim(self.batch_size, self.tp_conv[0]),
                          get_grid_dim(self.n_filters2, self.tp_conv[1]),
                          get_grid_dim(dims['p2_h'] * dims['p2_w'], self.tp_conv[2]))

        # For Dense layers (2D grid: B, OutFeatures)
        threads_per_block_2d = (16, 16) # 256 threads
        self.tp_dense = threads_per_block_2d
        self.blocks_hid = (get_grid_dim(self.batch_size, self.tp_dense[0]),
                           get_grid_dim(self.hidden, self.tp_dense[1]))

        # For Softmax/Backprop (1D grid: B)
        threads_per_block_1d = 256
        self.tp_1d = threads_per_block_1d
        self.grid_1d = get_grid_dim(self.batch_size, self.tp_1d)

        print("CUDA launch configurations defined.")
        print(f"Conv/Pool TPB: {self.tp_conv}, Dense TPB: {self.tp_dense}, 1D TPB: {self.tp_1d}")


    def train_epoch(self, x_train, y_train):
        """Trains the model for one epoch on CUDA."""
        n_train = len(x_train)
        total_loss = 0.0
        correct_preds = 0

        # Shuffle data on CPU
        permutation = np.random.permutation(n_train)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        # Process data in batches
        for i in range(0, n_train, self.batch_size):
            # --- Prepare batch ---
            batch_indices = np.arange(i, i + self.batch_size)
            xb_host = x_train_shuffled[batch_indices]
            yb_host = y_train_shuffled[batch_indices]

            # Copy batch data from Host (CPU) to Device (GPU)
            self.d_xb.copy_to_device(xb_host)
            self.d_yb.copy_to_device(yb_host)

            # --- Forward Pass (Execute GPU Kernels) ---
            conv2d_relu_cuda[self.blocks_c1, self.tp_conv](
                self.d_xb, self.d_weights['Wc1'], self.d_weights['bc1'], self.d_conv1_out)
            maxpool_cuda[self.blocks_p1, self.tp_conv](
                self.d_conv1_out, self.d_pool1_out)
            conv2d_relu_cuda[self.blocks_c2, self.tp_conv](
                self.d_pool1_out, self.d_weights['Wc2'], self.d_weights['bc2'], self.d_conv2_out)
            maxpool_cuda[self.blocks_p2, self.tp_conv](
                self.d_conv2_out, self.d_pool2_out)

            # Flatten - Reshape the pooling output view into the flat buffer
            cuda.synchronize() # Ensure pooling is done before reshape
            # Reshape directly into the allocated buffer (might involve copy internally)
            # Ensure flat_size matches the shape after pooling
            expected_flat_size = self.n_filters2 * self.dims['p2_h'] * self.dims['p2_w']
            if self.d_pool2_out.size != self.batch_size * expected_flat_size:
                 raise ValueError(f"Shape mismatch before flatten: pool2_out size {self.d_pool2_out.size} != expected {self.batch_size * expected_flat_size}")
            self.d_flat_out = self.d_pool2_out.reshape(self.batch_size, self.flat_size)


            dense_relu_cuda[self.blocks_hid, self.tp_dense](
                self.d_flat_out, self.d_weights['W1'], self.d_weights['b1'], self.d_hid_out)
            # Use 1D grid for softmax kernel
            dense_softmax_cuda[self.grid_1d, self.tp_1d](
                self.d_hid_out, self.d_weights['W2'], self.d_weights['b2'], self.d_logits, self.d_probs)

            # --- Calculate Loss & Accuracy (on CPU) ---
            # Copy probabilities back from Device (GPU) to Host (CPU)
            probs_host = self.d_probs.copy_to_host()

            # Calculate loss using NumPy on CPU
            log_probs = -np.log(probs_host[np.arange(self.batch_size), yb_host] + 1e-9)
            batch_loss = np.sum(log_probs)
            total_loss += batch_loss

            # Calculate accuracy using NumPy on CPU
            preds = np.argmax(probs_host, axis=1)
            correct_preds += np.sum(preds == yb_host)

            # --- Backward Pass ---
            # 1. Calculate output gradient (dL/dlogits) on GPU
            backprop_output_cuda[self.grid_1d, self.tp_1d](self.d_probs, self.d_yb, self.d_dlogits)

            # 2. FC Gradient Calculation & Weight Update (on CPU, as per example)
            # Copy necessary data back to host: dL/dlogits, hidden activations, flattened input
            dlog_host = self.d_dlogits.copy_to_host()
            hid_host = self.d_hid_out.copy_to_host()
            flat_host = self.d_flat_out.copy_to_host() # Copy the flattened data

            # Calculate gradients using NumPy (on CPU)
            grad_W2 = hid_host.T @ dlog_host
            grad_b2 = np.sum(dlog_host, axis=0)

            # Backpropagate gradient through W2 (on CPU)
            W2_host = self.weights_host['W2'] # Use the CPU copy of W2
            dh = dlog_host @ W2_host.T
            dh[hid_host <= 0] = 0 # Apply ReLU derivative based on activation values

            grad_W1 = flat_host.T @ dh
            grad_b1 = np.sum(dh, axis=0)

            # 3. Weight Update (on CPU using momentum)
            # Update velocities stored on host
            self.velocities_host['vW2'] = self.momentum * self.velocities_host['vW2'] + (1.0 - self.momentum) * grad_W2
            self.velocities_host['vb2'] = self.momentum * self.velocities_host['vb2'] + (1.0 - self.momentum) * grad_b2
            self.velocities_host['vW1'] = self.momentum * self.velocities_host['vW1'] + (1.0 - self.momentum) * grad_W1
            self.velocities_host['vb1'] = self.momentum * self.velocities_host['vb1'] + (1.0 - self.momentum) * grad_b1

            # Update weights stored on host
            self.weights_host['W2'] -= self.lr * self.velocities_host['vW2']
            self.weights_host['b2'] -= self.lr * self.velocities_host['vb2']
            self.weights_host['W1'] -= self.lr * self.velocities_host['vW1']
            self.weights_host['b1'] -= self.lr * self.velocities_host['vb1']

            # --- Copy updated FC weights back to GPU ---
            # This is a bottleneck compared to full GPU backprop.
            self.d_weights['W1'].copy_to_device(self.weights_host['W1'])
            self.d_weights['b1'].copy_to_device(self.weights_host['b1'])
            self.d_weights['W2'].copy_to_device(self.weights_host['W2'])
            self.d_weights['b2'].copy_to_device(self.weights_host['b2'])

            # Conv weights are not updated in this simplified backprop approach.

            # Synchronize device? Optional, might help timing consistency.
            # cuda.synchronize()

        avg_loss = total_loss / n_train if n_train > 0 else 0
        accuracy = correct_preds / n_train if n_train > 0 else 0
        return avg_loss, accuracy


    def evaluate(self, x_test, y_test):
        """Evaluates the model on the test set using CUDA."""
        n_test = len(x_test)
        if n_test == 0: return 0.0 # Handle empty test set
        correct_preds = 0

        # Process test data in batches
        for i in range(0, n_test, self.batch_size):
            # Prepare batch
            batch_indices = np.arange(i, min(i + self.batch_size, n_test))
            current_batch_size = len(batch_indices)
            if current_batch_size == 0: continue

            xb_host = x_test[batch_indices]
            yb_host = y_test[batch_indices] # Ground truth labels for this batch

            # --- Adjust for potential partial last batch ---
            # We need to handle the case where current_batch_size != self.batch_size
            # Pad the last batch (done in preprocessing)
            if current_batch_size != self.batch_size:
                 print(f"Warning: Test batch size mismatch ({current_batch_size} vs {self.batch_size}). Assuming test set was padded.")
                 pass


            # Copy input data to GPU
            self.d_xb.copy_to_device(xb_host)


            # --- Forward Pass (identical to training, no backprop needed) ---
            conv2d_relu_cuda[self.blocks_c1, self.tp_conv](
                self.d_xb, self.d_weights['Wc1'], self.d_weights['bc1'], self.d_conv1_out)
            maxpool_cuda[self.blocks_p1, self.tp_conv](
                self.d_conv1_out, self.d_pool1_out)
            conv2d_relu_cuda[self.blocks_c2, self.tp_conv](
                self.d_pool1_out, self.d_weights['Wc2'], self.d_weights['bc2'], self.d_conv2_out)
            maxpool_cuda[self.blocks_p2, self.tp_conv](
                self.d_conv2_out, self.d_pool2_out)

            cuda.synchronize()
            self.d_flat_out = self.d_pool2_out.reshape(self.batch_size, self.flat_size)

            dense_relu_cuda[self.blocks_hid, self.tp_dense](
                self.d_flat_out, self.d_weights['W1'], self.d_weights['b1'], self.d_hid_out)
            dense_softmax_cuda[self.grid_1d, self.tp_1d](
                self.d_hid_out, self.d_weights['W2'], self.d_weights['b2'], self.d_logits, self.d_probs)

            # --- Get predictions (on CPU) ---
            # Copy probabilities back to host
            probs_host = self.d_probs.copy_to_host()

            # Get predictions for the current batch
            preds = np.argmax(probs_host, axis=1)
            correct_preds += np.sum(preds == yb_host)

        accuracy = correct_preds / n_test # Use original number of test samples if unpadded
        return accuracy

