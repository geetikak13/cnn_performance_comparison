# main.py
import argparse
import numpy as np
import time
import torch # Needed for device checks and PyTorch model run
import os

# Project specific imports
# Ensure the script can find modules in src and data directories
import sys
sys.path.append(os.path.dirname(__file__)) # Add project root to path

from data.preprocess import load_mnist_numpy, load_mnist_pytorch
from src.utils import calculate_dims, initialize_cnn_weights_numpy
from src.logger import PerformanceLogger

# Platform specific runners/functions/classes
from src.model_cpu_numba import train_epoch_cpu, evaluate_cpu
# Import the runner class for CUDA
from src.model_cuda_numba import CUDAModelRunner
# Import PyTorch model and functions (now generalized)
from src.model_pytorch import ConvNetPyTorch, train_epoch_pytorch, evaluate_pytorch # Updated import
import torch.optim as optim
import torch.nn as nn


# ==============================================================================
# Experiment Runner Functions (Unchanged from previous version)
# ==============================================================================

def run_cpu_numba_experiment(params, logger): # Renamed for clarity
    """Runs the CPU (Numba) experiment."""
    platform_name = "CPU_Numba"
    logger.start_experiment(platform_name, device='cpu') # Log device as 'cpu'

    # Load data in NumPy format suitable for Numba
    try:
        (x_train, y_train), (x_test, y_test) = load_mnist_numpy(params['batch_size'])
    except Exception as e:
        print(f"Failed to load data for {platform_name} experiment: {e}")
        logger.end_experiment() # Mark experiment as ended even if failed
        return
    n_train = len(x_train)
    n_test = len(x_test) # Use original test size if needed for accuracy reporting

    # Calculate dimensions and initialize weights/velocities on CPU
    dims = calculate_dims(
        n_filters1=params['n_filters1'], n_filters2=params['n_filters2'],
        fsize=params['fsize'], hidden=params['hidden'],
        h0=x_train.shape[2], w0=x_train.shape[3] # Pass actual H, W
    )
    weights, velocities = initialize_cnn_weights_numpy(params, dims)

    # --- Training Loop ---
    try:
        for ep in range(1, params['epochs'] + 1):
            logger.start_epoch()
            train_loss, train_acc = train_epoch_cpu(x_train, y_train, weights, velocities, params, dims)
            # Evaluate on test set after each epoch
            test_acc = evaluate_cpu(x_test, y_test, weights, params, dims)
            logger.log_epoch(
                epoch=ep, train_loss=train_loss, train_acc=train_acc,
                test_acc=test_acc, num_train_samples=n_train
            )
    except Exception as e:
        print(f"An error occurred during {platform_name} training/evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.end_experiment()


def run_cuda_numba_experiment(params, logger): # Renamed for clarity
    """Runs the CUDA (Numba) experiment."""
    platform_name = "CUDA_Numba"
    logger.start_experiment(platform_name, device='cuda') # Log device as 'cuda'

    # Check for Numba CUDA availability
    try:
        from numba import cuda
        if not cuda.is_available():
             print("CUDA not available via Numba. Skipping CUDA (Numba) experiment.")
             logger.end_experiment()
             return
        print("Numba CUDA detected:")
        cuda.detect() # Print GPU info
    except ImportError:
        print("Numba CUDA extensions not installed or found. Skipping CUDA (Numba) experiment.")
        print("Try: conda install numba cudatoolkit=<your_cuda_version>")
        logger.end_experiment()
        return
    except Exception as e:
        print(f"Error initializing Numba CUDA: {e}. Skipping CUDA (Numba) experiment.")
        logger.end_experiment()
        return

    # Load data in NumPy format
    try:
        (x_train, y_train), (x_test, y_test) = load_mnist_numpy(params['batch_size'])
    except Exception as e:
        print(f"Failed to load data for {platform_name} experiment: {e}")
        logger.end_experiment()
        return
    n_train = len(x_train)
    n_test = len(x_test)

    # Calculate dimensions and initialize weights/velocities on CPU first
    dims = calculate_dims(
        n_filters1=params['n_filters1'], n_filters2=params['n_filters2'],
        fsize=params['fsize'], hidden=params['hidden'], h0=x_train.shape[2], w0=x_train.shape[3]
    )
    initial_weights, initial_velocities = initialize_cnn_weights_numpy(params, dims)

    # Initialize CUDA runner (handles GPU resources and kernel calls)
    try:
        cuda_runner = CUDAModelRunner(params, dims, initial_weights, initial_velocities)
    except Exception as e:
        print(f"Error initializing CUDAModelRunner: {e}")
        logger.end_experiment()
        return

    # --- Training Loop ---
    try:
        for ep in range(1, params['epochs'] + 1):
            logger.start_epoch()
            train_loss, train_acc = cuda_runner.train_epoch(x_train, y_train)
            # Evaluate on test set
            test_acc = cuda_runner.evaluate(x_test, y_test)
            logger.log_epoch(
                epoch=ep, train_loss=train_loss, train_acc=train_acc,
                test_acc=test_acc, num_train_samples=n_train
            )
    except Exception as e:
        print(f"An error occurred during {platform_name} training/evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up GPU memory? Numba might handle this, but explicit del could be added if needed.
        # del cuda_runner # Potentially release GPU resources
        # cuda.close() # Close context if needed
        logger.end_experiment()


def run_pytorch_experiment(params, logger, device_type):
    """Runs a PyTorch experiment on the specified device (cpu, cuda, mps)."""

    # Determine platform name and device object
    if device_type == 'cpu':
        platform_name = "PyTorch_CPU"
        device = torch.device("cpu")
        print("PyTorch using CPU device.")
    elif device_type == 'cuda':
        platform_name = "PyTorch_CUDA"
        if not torch.cuda.is_available():
            print("CUDA not available via PyTorch. Skipping PyTorch CUDA experiment.")
            return # Don't start/end experiment if device unavailable
        device = torch.device("cuda")
        print(f"PyTorch using CUDA device: {torch.cuda.get_device_name(device)}")
    elif device_type == 'mps':
        platform_name = "PyTorch_MPS"
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("MPS not available via PyTorch. Skipping PyTorch MPS experiment.")
            return # Don't start/end experiment if device unavailable
        device = torch.device("mps")
        print("PyTorch using MPS device.")
    else:
        print(f"Error: Invalid device type '{device_type}' for PyTorch experiment.")
        return

    logger.start_experiment(platform_name, device=device_type)

    # Load data using PyTorch DataLoader
    try:
        # Use the same batch size for fair comparison, adjust if GPU runs out of memory
        train_loader, test_loader = load_mnist_pytorch(batch_size=params['batch_size'])
    except Exception as e:
        print(f"Failed to load data for {platform_name} experiment: {e}")
        logger.end_experiment()
        return

    # Get number of training samples (may include padding)
    n_train = len(train_loader.dataset)

    # Calculate dimensions needed for model init
    # Get H, W from a sample batch if possible
    try:
        sample_data, _ = next(iter(train_loader))
        _, _, h0, w0 = sample_data.shape
    except Exception:
        print("Warning: Could not get dimensions from DataLoader, using default 28x28.")
        h0, w0 = 28, 28

    dims = calculate_dims(
        n_filters1=params['n_filters1'], n_filters2=params['n_filters2'],
        fsize=params['fsize'], hidden=params['hidden'], h0=h0, w0=w0
    )

    # Initialize PyTorch model and move to the selected device
    model = ConvNetPyTorch(params, dims).to(device)

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    try:
        for ep in range(1, params['epochs'] + 1):
            logger.start_epoch()
            train_loss, train_acc = train_epoch_pytorch(model, device, train_loader, optimizer, criterion)
            # Evaluate on test set
            _, test_acc = evaluate_pytorch(model, device, test_loader, criterion) # evaluate returns (loss, acc)
            logger.log_epoch(
                epoch=ep, train_loss=train_loss, train_acc=train_acc,
                test_acc=test_acc, num_train_samples=n_train
            )
            # Clear CUDA cache periodically if using CUDA to potentially free memory
            if device_type == 'cuda':
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred during {platform_name} training/evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.end_experiment()


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run CNN Performance Comparison Experiments")
    # Updated platform choices
    parser.add_argument('--platform', type=str, default='all',
                        choices=['cpu_numba', 'cuda_numba',
                                 'pytorch_cpu', 'pytorch_cuda', 'pytorch_mps',
                                 'all'],
                        help='Platform(s) to run experiment on')
    # Allow overriding default hyperparameters from examples
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD (default: 0.9)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--filters1', type=int, default=8, help='Number of filters in Conv Layer 1 (default: 8)')
    parser.add_argument('--filters2', type=int, default=16, help='Number of filters in Conv Layer 2 (default: 16)')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units in FC Layer 1 (default: 128)')
    parser.add_argument('--fsize', type=int, default=3, help='Convolutional filter size (default: 3)')
    # Changed log_file to log_dir
    parser.add_argument('--log_dir', type=str, default='results', help='Directory to save metrics CSV files (default: results)')

    args = parser.parse_args()

    # Store parameters in a dictionary
    hyperparameters = {
        'lr': args.lr,
        'momentum': args.momentum,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'n_filters1': args.filters1,
        'n_filters2': args.filters2,
        'hidden': args.hidden,
        'fsize': args.fsize
    }

    print("Starting CNN Performance Comparison Experiments")
    print("Parameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print(f"Platform(s): {args.platform}")
    # Updated log message
    print(f"Logging to directory: {args.log_dir}")

    # Initialize logger with the specified directory
    perf_logger = PerformanceLogger(log_directory=args.log_dir)

    # --- Run Selected Experiments ---
    start_overall_time = time.time()

    # Define which platforms to run based on 'all' flag or specific choice
    platforms_to_run = []
    if args.platform == 'all':
        platforms_to_run = ['cpu_numba', 'cuda_numba', 'pytorch_cpu', 'pytorch_cuda', 'pytorch_mps']
    else:
        platforms_to_run = [args.platform]

    # Execute experiments based on the list
    if 'cpu_numba' in platforms_to_run:
        run_cpu_numba_experiment(hyperparameters, perf_logger)

    if 'cuda_numba' in platforms_to_run:
        run_cuda_numba_experiment(hyperparameters, perf_logger)

    if 'pytorch_cpu' in platforms_to_run:
        run_pytorch_experiment(hyperparameters, perf_logger, device_type='cpu')

    if 'pytorch_cuda' in platforms_to_run:
        run_pytorch_experiment(hyperparameters, perf_logger, device_type='cuda')

    if 'pytorch_mps' in platforms_to_run:
        run_pytorch_experiment(hyperparameters, perf_logger, device_type='mps')


    end_overall_time = time.time()
    print("\n--- All specified experiments complete ---")
    print(f"Total execution time: {end_overall_time - start_overall_time:.2f}s")
    # Updated log message
    print(f"Results logged to individual files in '{perf_logger.log_directory}' directory.")
    print("Run 'python evaluation/compare_results.py' to analyze and visualize combined results.")

