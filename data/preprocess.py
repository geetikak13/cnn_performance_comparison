# data/preprocess.py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tensorflow.keras.datasets import mnist
import ssl # Import ssl module
import os

# --- Temporary SSL Workaround ---
# Problem: Sometimes Python cannot find SSL certificates, causing download errors.
# Solution: Temporarily disable SSL verification for downloads.
# WARNING: This is less secure. The preferred solution is to fix the system's
# certificate setup (e.g., run 'Install Certificates.command' on macOS).
# Set this flag to True ONLY if you encounter SSL errors and cannot fix them system-wide.
ALLOW_UNVERIFIED_SSL = True # <-- SET TO True TO TRY WORKAROUND, False otherwise

if ALLOW_UNVERIFIED_SSL and hasattr(ssl, '_create_unverified_context'):
    print("Warning: Applying temporary SSL workaround (disabling certificate verification).")
    # Monkey-patch the SSL context creation for torchvision download
    ssl._create_default_https_context = ssl._create_unverified_context
# --- End SSL Workaround ---


def load_mnist_numpy(batch_size: int):
    """
    Loads and preprocesses MNIST data into NumPy arrays suitable for Numba.
    Uses tf.keras.datasets.mnist as shown in the example files.
    Adds padding to make dataset size a multiple of batch_size.
    Adds a channel dimension.

    Args:
        batch_size (int): The batch size for training and testing.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
               NumPy arrays for training and testing data and labels.
               x_train/x_test shape: (N, 1, H, W)
    """
    print("Loading MNIST data using TensorFlow Keras API...")
    # Note: The SSL workaround above might not affect TensorFlow's download mechanism.
    # If TF download fails, ensure certificates are fixed system-wide or download manually.
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception as e:
        print(f"Error loading MNIST data via TensorFlow: {e}")
        if isinstance(e, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(e):
             print("\nSSL Error detected during TensorFlow download.")
             print("The temporary SSL workaround might not apply here.")
             print("Please try fixing system certificates (e.g., run 'Install Certificates.command' on macOS).")
        else:
            print("Please ensure TensorFlow is installed (`pip install tensorflow`)")
            print("or check your internet connection.")
        raise

    # Normalize and type cast
    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.astype(np.int32)

    # Add channel dimension for Numba implementations (N, H, W) -> (N, 1, H, W)
    x_train = x_train[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]

    n_train_orig, n_test_orig = len(x_train), len(x_test)

    # Pad to full batches
    pad_train = (batch_size - (n_train_orig % batch_size)) % batch_size
    if pad_train > 0:
        print(f"Padding training data with {pad_train} samples.")
        # Repeat samples from the beginning to pad
        x_train = np.vstack([x_train, x_train[:pad_train]])
        y_train = np.hstack([y_train, y_train[:pad_train]])

    pad_test = (batch_size - (n_test_orig % batch_size)) % batch_size
    if pad_test > 0:
        print(f"Padding test data with {pad_test} samples.")
        # Repeat samples from the beginning to pad
        x_test = np.vstack([x_test, x_test[:pad_test]])
        y_test = np.hstack([y_test, y_test[:pad_test]])

    n_train, n_test = len(x_train), len(x_test)
    print(f"Loaded MNIST via TF: {n_train_orig} train -> {n_train} padded, "
          f"{n_test_orig} test -> {n_test} padded.")
    print(f"Shapes: x_train={x_train.shape}, y_train={y_train.shape}, "
          f"x_test={x_test.shape}, y_test={y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def load_mnist_pytorch(batch_size: int):
    """
    Loads and preprocesses MNIST using PyTorch's torchvision.
    Handles padding to ensure dataset size is a multiple of batch_size.

    Args:
        batch_size (int): The batch size for the DataLoader.

    Returns:
        tuple: (train_loader, test_loader)
               PyTorch DataLoaders for training and testing.
    """
    print("Loading MNIST data using Torchvision...")
    # Standard transformations for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL image or numpy array to tensor (C, H, W) and scales to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std dev of MNIST dataset
    ])

    # Define dataset root directory
    data_root = './data'
    os.makedirs(data_root, exist_ok=True) # Ensure data directory exists

    try:
        # Download dataset if not present - SSL workaround applies here if enabled
        trainset = torchvision.datasets.MNIST(root=data_root, train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_root, train=False,
                                             download=True, transform=transform)
    except Exception as e:
        print(f"Error loading/downloading MNIST via Torchvision: {e}")
        if isinstance(e, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(e):
             print("\nSSL Error detected during Torchvision download.")
             if ALLOW_UNVERIFIED_SSL:
                 print("The temporary SSL workaround was applied but might not have been sufficient.")
             else:
                 print("Try setting ALLOW_UNVERIFIED_SSL = True in preprocess.py as a temporary test,")
                 print("or fix system certificates (e.g., run 'Install Certificates.command' on macOS).")
        else:
            print("Please check your internet connection or file permissions for the './data' directory.")
        raise


    n_train_orig = len(trainset)
    n_test_orig = len(testset)

    # --- Padding Logic for DataLoader ---
    # Calculate padding needed
    pad_train = (batch_size - (n_train_orig % batch_size)) % batch_size
    pad_test = (batch_size - (n_test_orig % batch_size)) % batch_size

    if pad_train > 0:
        print(f"Padding training data with {pad_train} samples for PyTorch DataLoader.")
        # Create a dataset of indices to repeat from the original dataset
        padding_indices = list(range(pad_train))
        # Create a Subset using these indices and concatenate it
        padding_dataset = torch.utils.data.Subset(trainset, padding_indices)
        padded_trainset = torch.utils.data.ConcatDataset([trainset, padding_dataset])
    else:
        padded_trainset = trainset

    if pad_test > 0:
        print(f"Padding test data with {pad_test} samples for PyTorch DataLoader.")
        padding_indices = list(range(pad_test))
        padding_dataset = torch.utils.data.Subset(testset, padding_indices)
        padded_testset = torch.utils.data.ConcatDataset([testset, padding_dataset])
    else:
        padded_testset = testset

    n_train_padded = len(padded_trainset)
    n_test_padded = len(padded_testset)

    # Create DataLoaders
    # num_workers > 0 can speed up data loading but might cause issues on some systems/OS
    # Use pin_memory=True if using CUDA for faster host-to-device transfers
    # Set num_workers=0 if you suspect multi-processing issues
    num_workers = 2 if os.name != 'nt' else 0 # Often safer to use 0 on Windows
    train_loader = torch.utils.data.DataLoader(padded_trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = torch.utils.data.DataLoader(padded_testset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"Created PyTorch DataLoaders: "
          f"{n_train_orig} train -> {n_train_padded} padded, "
          f"{n_test_orig} test -> {n_test_padded} padded.")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    return train_loader, test_loader


if __name__ == '__main__':
    # Example usage:
    print("--- Testing NumPy Loader ---")
    try:
        (xtr_np, ytr_np), (xte_np, yte_np) = load_mnist_numpy(batch_size=512)
    except Exception as e:
        print(f"NumPy Loader test failed: {e}")


    print("\n--- Testing PyTorch Loader ---")
    try:
        train_loader_pt, test_loader_pt = load_mnist_pytorch(batch_size=64)
        # Accessing a batch to check shapes
        images, labels = next(iter(train_loader_pt))
        print("PyTorch batch shapes:", images.shape, labels.shape) # Should be [64, 1, 28, 28], [64]
    except Exception as e:
         print(f"PyTorch Loader test failed: {e}")
