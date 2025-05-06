# src/model_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from .utils import calculate_dims # To get flat_size dynamically

# ==============================================================================
# PyTorch CNN Model Definition
# ==============================================================================

class ConvNetPyTorch(nn.Module):
    """
    Defines the CNN architecture using PyTorch modules.
    Matches the structure used in the Numba implementations.
    Can run on CPU, CUDA, or MPS devices.
    """
    def __init__(self, params, dims):
        """
        Initializes the layers of the CNN.

        Args:
            params (dict): Dictionary containing hyperparameters like n_filters1, n_filters2, hidden, fsize.
            dims (dict): Dictionary containing calculated dimensions like flat_size.
        """
        super(ConvNetPyTorch, self).__init__() # Use new class name
        n_filters1 = params['n_filters1']
        n_filters2 = params['n_filters2']
        fsize = params['fsize']
        hidden = params['hidden']
        flat_size = dims['flat_size'] # Get calculated flat size

        # Convolutional Layer 1: Input channels = 1 (MNIST is grayscale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters1, kernel_size=fsize, stride=1, padding=0)
        # Max Pooling Layer 1: Kernel size = 2, Stride = 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional Layer 2: Input channels = n_filters1
        self.conv2 = nn.Conv2d(in_channels=n_filters1, out_channels=n_filters2, kernel_size=fsize, stride=1, padding=0)
        # Max Pooling Layer 2: Kernel size = 2, Stride = 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dense (Fully Connected) Layer 1: Input features = flat_size
        self.fc1 = nn.Linear(in_features=flat_size, out_features=hidden)
        # Dense (Fully Connected) Layer 2 (Output Layer): Input features = hidden
        self.fc2 = nn.Linear(in_features=hidden, out_features=10) # Output size 10 for MNIST

        print(f"PyTorch {self.__class__.__name__} model initialized:") # Use dynamic class name
        print(self) # Print model summary


    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor (batch of images). Shape (B, 1, H, W).

        Returns:
            torch.Tensor: Output tensor (logits). Shape (B, 10).
        """
        # Apply Conv1 -> ReLU -> Pool1
        x = self.pool1(F.relu(self.conv1(x)))
        # Apply Conv2 -> ReLU -> Pool2
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten the output from the pooling layer
        x = x.view(x.shape[0], -1) # Reshape to (BatchSize, flat_size)
        # Apply FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # Apply FC2 (Output Layer) - return logits
        x = self.fc2(x)
        return x

# ==============================================================================
# PyTorch Training and Evaluation Functions (Device Agnostic)
# ==============================================================================

def train_epoch_pytorch(model, device, train_loader, optimizer, criterion):
    """
    Trains the PyTorch model for one epoch on the specified device (CPU/CUDA/MPS).

    Args:
        model (nn.Module): The PyTorch model.
        device (torch.device): The device to train on (e.g., 'cpu', 'cuda', 'mps').
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): The optimizer (e.g., SGD, Adam).
        criterion (Loss): The loss function (e.g., CrossEntropyLoss).

    Returns:
        tuple: (average_loss, accuracy) for the epoch.
    """
    model.train() # Set the model to training mode
    total_loss = 0.0
    correct_preds = 0
    num_samples = 0 # Keep track of total samples processed

    # Iterate over batches from the DataLoader
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target tensors to the designated device
        data, target = data.to(device), target.to(device)
        current_batch_size = data.size(0)
        num_samples += current_batch_size

        # --- Forward Pass ---
        optimizer.zero_grad() # Clear gradients
        output = model(data)  # Get model predictions (logits)

        # --- Calculate Loss ---
        loss = criterion(output, target)

        # --- Backward Pass ---
        loss.backward() # Compute gradients

        # --- Optimizer Step ---
        optimizer.step() # Update parameters

        # --- Accumulate Metrics ---
        total_loss += loss.item() * current_batch_size # Accumulate total loss
        preds = output.argmax(dim=1, keepdim=True)
        correct_preds += preds.eq(target.view_as(preds)).sum().item()

    # Calculate average loss and accuracy for the entire epoch
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    accuracy = correct_preds / num_samples if num_samples > 0 else 0
    return avg_loss, accuracy


def evaluate_pytorch(model, device, test_loader, criterion):
    """
    Evaluates the PyTorch model on the test set using the specified device (CPU/CUDA/MPS).

    Args:
        model (nn.Module): The PyTorch model.
        device (torch.device): The device to evaluate on.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (Loss): The loss function (used for reporting test loss).

    Returns:
        tuple: (average_test_loss, test_accuracy)
    """
    model.eval() # Set the model to evaluation mode
    test_loss = 0.0
    correct_preds = 0
    num_samples = 0

    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            current_batch_size = data.size(0)
            num_samples += current_batch_size

            output = model(data) # Forward pass

            # Calculate and sum up batch loss
            test_loss += criterion(output, target).item() * current_batch_size

            # Get predictions and count correct ones
            preds = output.argmax(dim=1, keepdim=True)
            correct_preds += preds.eq(target.view_as(preds)).sum().item()

    # Calculate average loss and accuracy over the entire test set
    avg_test_loss = test_loss / num_samples if num_samples > 0 else 0
    accuracy = correct_preds / num_samples if num_samples > 0 else 0

    return avg_test_loss, accuracy


if __name__ == '__main__':
    print("--- Testing Generic PyTorch Model ---")
    # Dummy parameters and dimensions for testing model instantiation
    test_params = {'n_filters1': 8, 'n_filters2': 16, 'fsize': 3, 'hidden': 128, 'lr': 0.01, 'momentum': 0.9, 'batch_size': 64, 'epochs': 1}
    test_dims = calculate_dims()

    # Test device selection (CPU by default if others fail)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Testing on CUDA device.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Testing on MPS device.")
    else:
        device = torch.device("cpu")
        print("Testing on CPU device.")

    # Instantiate model
    model = ConvNetPyTorch(test_params, test_dims).to(device)

    # Test with dummy data
    print("\nTesting forward pass...")
    dummy_input = torch.randn(test_params['batch_size'], 1, 28, 28).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [batch_size, 10]
    assert output.shape == (test_params['batch_size'], 10)

    # --- Optional: Test training/evaluation loop with dummy data ---
    # (Requires data loader setup as before)
