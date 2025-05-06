# src/model_cpu_numba.py
import numpy as np
from numba import njit, prange
import math
import time

# --- Re-use kernels from cpu_impl.py ---
@njit(parallel=True, fastmath=True)
def conv2d_relu_cpu(inp, W, b, out):
    B, C, H, W_in = inp.shape
    F, _, f, _ = W.shape
    oh, ow = H-f+1, W_in-f+1
    for bi in prange(B): # Parallelize over batch
        for fo in range(F):
            for oy in range(oh):
                for ox in range(ow):
                    acc = b[fo]
                    for ci in range(C):
                        for u in range(f):
                            for v in range(f):
                                acc += inp[bi,ci,oy+u,ox+v] * W[fo,ci,u,v]
                    out[bi,fo,oy,ox] = acc if acc>0 else 0.0 # ReLU

@njit(parallel=True, fastmath=True)
def maxpool_cpu(inp, out):
    B,C,H,W_in = inp.shape
    oh, ow = H//2, W_in//2
    for bi in prange(B): # Parallelize over batch
        for ci in range(C):
            for oy in range(oh):
                for ox in range(ow):
                    # Find max in 2x2 window
                    i0 = inp[bi,ci,oy*2,  ox*2]
                    i1 = inp[bi,ci,oy*2,  ox*2+1]
                    i2 = inp[bi,ci,oy*2+1,ox*2]
                    i3 = inp[bi,ci,oy*2+1,ox*2+1]
                    m = i0 if i0>i1 else i1
                    m = i2 if i2>m  else m
                    out[bi,ci,oy,ox] = i3 if i3>m  else m

@njit(parallel=True, fastmath=True)
def dense_relu_cpu(inp, W, b, out):
    B,D = inp.shape
    _, K = W.shape
    for i in prange(B): # Parallelize over batch
        for j in range(K):
            acc = b[j]
            for k in range(D):
                acc += inp[i,k]*W[k,j]
            out[i,j] = acc if acc>0 else 0.0 # ReLU

@njit(parallel=True, fastmath=True)
def dense_softmax_cpu(inp, W, b, logits, probs):
    B,D = inp.shape
    _, K = W.shape
    for i in prange(B): # Parallelize over batch
        # Calculate logits
        max_logit_in_row = -np.inf # For numerical stability
        for j in range(K):
            acc = b[j]
            for k in range(D):
                acc += inp[i,k]*W[k,j]
            logits[i,j] = acc
            if acc > max_logit_in_row:
                max_logit_in_row = acc

        # Calculate softmax probabilities
        sum_exp = 0.0
        for j in range(K):
            exp_val = math.exp(logits[i,j] - max_logit_in_row) # Subtract max logit
            probs[i,j] = exp_val
            sum_exp += exp_val

        # Normalize
        inv_sum_exp = 1.0 / sum_exp if sum_exp > 0 else 1.0
        for j in range(K):
            probs[i,j] *= inv_sum_exp


@njit(parallel=True) # No fastmath needed here
def backprop_output_cpu(probs, y, dlogits):
    B,K = probs.shape
    for i in prange(B): # Parallelize over batch
        yi = y[i]
        for j in range(K):
            # Gradient of Softmax + CrossEntropyLoss
            dlogits[i,j] = probs[i,j] - (1.0 if j == yi else 0.0)


# --- Backpropagation Kernels ---
@njit(parallel=True, fastmath=True)
def backprop_dense_cpu(d_out, W, activation_in, d_inp):
    # d_out: gradient from the next layer (e.g., dlogits)
    # W: weights of the current dense layer
    # activation_in: input to the current dense layer's activation (e.g., hid_out for W2)
    # d_inp: gradient w.r.t the input of the current dense layer (output)
    B, K_out = d_out.shape
    D_in, _ = W.shape # W shape is (D_in, K_out)

    for i in prange(B): # Parallelize over batch
        for k in range(D_in):
            acc = 0.0
            for j in range(K_out):
                 acc += d_out[i, j] * W[k, j]
            # Apply derivative of ReLU (1 if input > 0, else 0)
            d_inp[i, k] = acc * (1.0 if activation_in[i, k] > 0 else 0.0)


@njit(parallel=True, fastmath=True)
def update_weights_momentum_cpu(W, b, grad_W, grad_b, vW, vb, lr, momentum):
    # Update velocities
    vW[:] = momentum * vW + (1.0 - momentum) * grad_W
    vb[:] = momentum * vb + (1.0 - momentum) * grad_b
    # Update weights
    W -= lr * vW
    b -= lr * vb

# --- Training and Evaluation Loop ---
def train_epoch_cpu(x_train, y_train, weights, velocities, params, dims):
    n_train = len(x_train)
    batch_size = params['batch_size']
    n_filters1 = params['n_filters1']
    n_filters2 = params['n_filters2']
    hidden = params['hidden']
    lr = params['lr']
    momentum = params['momentum']
    flat_size = dims['flat_size']

    Wc1, bc1 = weights['Wc1'], weights['bc1']
    Wc2, bc2 = weights['Wc2'], weights['bc2']
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    vWc1, vbc1 = velocities['vWc1'], velocities['vbc1']
    vWc2, vbc2 = velocities['vWc2'], velocities['vbc2']
    vW1, vb1 = velocities['vW1'], velocities['vb1']
    vW2, vb2 = velocities['vW2'], velocities['vb2']


    # Pre-allocate buffers for one batch
    conv1   = np.empty((batch_size, n_filters1, dims['c1_h'], dims['c1_w']), np.float32)
    pool1   = np.empty((batch_size, n_filters1, dims['p1_h'], dims['p1_w']), np.float32)
    conv2   = np.empty((batch_size, n_filters2, dims['c2_h'], dims['c2_w']), np.float32)
    pool2   = np.empty((batch_size, n_filters2, dims['p2_h'], dims['p2_w']), np.float32)
    flat    = np.empty((batch_size, flat_size), np.float32)
    hid_out = np.empty((batch_size, hidden), np.float32)
    logits  = np.empty((batch_size, 10), np.float32)
    probs   = np.empty((batch_size, 10), np.float32)
    dlogits = np.empty((batch_size, 10), np.float32)
    # Add buffers for backprop gradients if implementing conv backprop
    d_hid = np.empty_like(hid_out)

    total_loss = 0.0
    correct_preds = 0
    start_time = time.time()

    # Shuffle training data each epoch
    permutation = np.random.permutation(n_train)
    x_train_shuffled = x_train[permutation]
    y_train_shuffled = y_train[permutation]


    for bs in range(0, n_train, batch_size):
        xb = x_train_shuffled[bs : bs + batch_size]
        yb = y_train_shuffled[bs : bs + batch_size]

        # --- Forward Pass ---
        conv2d_relu_cpu(xb, Wc1, bc1, conv1)
        maxpool_cpu(conv1, pool1)
        conv2d_relu_cpu(pool1, Wc2, bc2, conv2)
        maxpool_cpu(conv2, pool2)

        flat_batch = pool2.reshape(batch_size, flat_size) # Use dedicated buffer if needed
        dense_relu_cpu(flat_batch, W1, b1, hid_out)
        dense_softmax_cpu(hid_out, W2, b2, logits, probs)

        # --- Calculate Loss & Accuracy ---
        # Cross-entropy loss (using stable log-sum-exp would be better, but matching example)
        log_probs = -np.log(probs[np.arange(batch_size), yb] + 1e-9) # Add epsilon for stability
        batch_loss = np.sum(log_probs)
        total_loss += batch_loss
        preds = np.argmax(probs, axis=1)
        correct_preds += np.sum(preds == yb)

        # --- Backward Pass ---
        backprop_output_cpu(probs, yb, dlogits) # Gradient for Softmax+Loss

        # Backprop through Dense Layer 2 (Output Layer)
        # Calculate gradients for W2, b2
        # Need hid_out.T @ dlogits which is tricky with njit parallel. Do it serially.
        grad_W2 = hid_out.T @ dlogits
        grad_b2 = np.sum(dlogits, axis=0)
        # Calculate gradient w.r.t hidden layer output (d_hid)
        backprop_dense_cpu(dlogits, W2, hid_out, d_hid) # Pass hid_out as activation_in

        # Backprop through Dense Layer 1 (Hidden Layer)
        # Calculate gradients for W1, b1
        grad_W1 = flat_batch.T @ d_hid
        grad_b1 = np.sum(d_hid, axis=0)
        # Calculate gradient w.r.t flattened layer output (if needed for conv backprop)
        # backprop_dense_cpu(d_hid, W1, flat_batch, d_flat) # Pass flat_batch as activation_in

        # --- Backprop ---
        # Updated FC layers.
        grad_Wc1, grad_bc1 = np.zeros_like(Wc1), np.zeros_like(bc1)
        grad_Wc2, grad_bc2 = np.zeros_like(Wc2), np.zeros_like(bc2)

        # --- Update Weights (only FC layers as per example) ---
        update_weights_momentum_cpu(W2, b2, grad_W2, grad_b2, vW2, vb2, lr, momentum)
        update_weights_momentum_cpu(W1, b1, grad_W1, grad_b1, vW1, vb1, lr, momentum)


    avg_loss = total_loss / n_train
    accuracy = correct_preds / n_train
    return avg_loss, accuracy


def evaluate_cpu(x_test, y_test, weights, params, dims):
    n_test = len(x_test)
    batch_size = params['batch_size']
    n_filters1 = params['n_filters1']
    n_filters2 = params['n_filters2']
    hidden = params['hidden']
    flat_size = dims['flat_size']

    Wc1, bc1 = weights['Wc1'], weights['bc1']
    Wc2, bc2 = weights['Wc2'], weights['bc2']
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']

    # Pre-allocate buffers for one batch
    conv1   = np.empty((batch_size, n_filters1, dims['c1_h'], dims['c1_w']), np.float32)
    pool1   = np.empty((batch_size, n_filters1, dims['p1_h'], dims['p1_w']), np.float32)
    conv2   = np.empty((batch_size, n_filters2, dims['c2_h'], dims['c2_w']), np.float32)
    pool2   = np.empty((batch_size, n_filters2, dims['p2_h'], dims['p2_w']), np.float32)
    flat    = np.empty((batch_size, flat_size), np.float32)
    hid_out = np.empty((batch_size, hidden), np.float32)
    logits  = np.empty((batch_size, 10), np.float32)
    probs   = np.empty((batch_size, 10), np.float32)

    correct_preds = 0
    for bs in range(0, n_test, batch_size):
        xb = x_test[bs : bs + batch_size]
        yb = y_test[bs : bs + batch_size]

        conv2d_relu_cpu(xb, Wc1, bc1, conv1)
        maxpool_cpu(conv1, pool1)
        conv2d_relu_cpu(pool1, Wc2, bc2, conv2)
        maxpool_cpu(conv2, pool2)

        flat_batch = pool2.reshape(batch_size, flat_size)
        dense_relu_cpu(flat_batch, W1, b1, hid_out)
        dense_softmax_cpu(hid_out, W2, b2, logits, probs)

        preds = np.argmax(probs, axis=1)
        correct_preds += np.sum(preds == yb)

    accuracy = correct_preds / n_test
    return accuracy