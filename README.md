# CNN Performance Comparison: CPU vs. CUDA vs. MPS vs. PyTorch Backends

## Project Goal

This project implements and compares a Convolutional Neural Network (CNN) for MNIST image classification across **five** different hardware/framework combinations:
1.  **CPU (Numba):** Using Numba's `njit` for Just-In-Time compilation and parallel execution on CPU cores.
2.  **CUDA GPU (Numba):** Using Numba's `cuda.jit` for NVIDIA GPU acceleration, with manual kernel implementation.
3.  **CPU (PyTorch):** Using the standard PyTorch framework explicitly targeting the CPU backend.
4.  **CUDA GPU (PyTorch):** Using the standard PyTorch framework explicitly targeting the CUDA backend for NVIDIA GPUs.
5.  **Apple Silicon GPU (PyTorch MPS):** Using PyTorch's Metal Performance Shaders (MPS) backend for Apple Silicon GPUs.

The primary objective is to quantify and compare the performance characteristics (training speed, throughput, accuracy, basic memory usage) of these different implementations based on the approach outlined in the initial project report.

## Architecture

The CNN architecture used is consistent across all implementations:
* Input: 28x28 MNIST images (normalized)
* Layer 1: Conv2D (1 input channel -> 8 filters, 3x3 kernel) + ReLU
* Layer 2: MaxPool2D (2x2 kernel, stride 2)
* Layer 3: Conv2D (8 input channels -> 16 filters, 3x3 kernel) + ReLU
* Layer 4: MaxPool2D (2x2 kernel, stride 2)
* Layer 5: Flatten
* Layer 6: Dense (Fully Connected) (Calculated input size -> 128 units) + ReLU
* Layer 7: Dense (Fully Connected) (128 units -> 10 units for MNIST classes) + Softmax (for Numba) / Logits (for PyTorch with CrossEntropyLoss)

Hyperparameters (based on example files, can be overridden via command line):
* Learning Rate: 0.01
* Momentum: 0.9 (for SGD optimizer)
* Batch Size: 512
* Epochs: 10

## Project Structure
```
cnn_performance_comparison/
├── .gitignore
├── README.md
├── Report/
│   ├── 605ProjectReport.pdf      # Final Project Report
│   ├── 605ProjectReport.tex      # Final Project Report latex
│   ├── Group-6 Final Report.pptx # Final Project Presentation 
│   ├── Group-6 Mid-Sem.pptx      # Mid Semester Project Presentation
├── data/
│   └── preprocess.py             # Load/preprocess MNIST data
├── src/
│   ├── model_cpu_numba.py        # CPU Numba implementation
│   ├── model_cuda_numba.py       # CUDA Numba implementation
│   ├── model_pytorch.py          # PyTorch model definition & train/eval loops
│   ├── logger.py                 # Metrics logging utility
│   └── utils.py                  # Common functions (e.g., weights init)
├── results/                      # Saved logs & plots (created on run)
│   └── metrics.csv               # Seaparate for each hardware combination
│   └── comparison_plot.png       # combined plot analysis
├── evaluation/
│   └── compare_results.py        # Analysis and visualization script
└── main.py                       # Main runner script
```

1.  **Clone the repository :**
    ```bash
    git clone https://github.com/geetikak13/cnn_performance_comparison
    cd model-pruning
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or use: python3 -m pip install -r requirements.txt
    ```
    *Note: This installs PyTorch, Torchvision, NumPy, Matplotlib, Pandas, and thop.*

## Dataset Download

The MNIST dataset is required. It will be automatically downloaded to the `./data` directory when first needed by any script.

## Requirements

* Python 3.8+
* NumPy (`pip install numpy`)
* Numba (`pip install numba` or `conda install numba`)
    * For CUDA (Numba): `conda install numba cudatoolkit=<version>` (match your driver version) or `pip install numba` (ensure CUDA toolkit is installed separately and in PATH)
* PyTorch (`pip install torch torchvision torchaudio` or follow PyTorch website instructions for specific CPU/CUDA/MPS versions)
* TensorFlow (`pip install tensorflow`) - *Only* for `tf.keras.datasets.mnist` as used in examples. Can be replaced with `torchvision.datasets.MNIST`. This implementation uses TF for consistency with examples.
* Matplotlib (`pip install matplotlib`)
* Seaborn (`pip install seaborn`)
* Pandas (`pip install pandas`)
* Psutil (`pip install psutil`) - For basic RAM logging.
* NVIDIA GPU with CUDA support and compatible drivers.
* Apple Silicon Mac with macOS 12.3+.

## Usage

1.  **Setup Environment:** Install the required libraries (e.g., using `pip` or `conda`).
2.  **Run Training & Evaluation:**
    Execute the main script from the `cnn_performance_comparison` directory. Choose the platform(s) to run using the `--platform` argument:
    ```bash
    # Run only the Numba CPU version
    python3 main.py --platform cpu_numba

    # Run only the Numba CUDA version
    python3 main.py --platform cuda_numba

    # Run only the PyTorch CPU version
    python3 main.py --platform pytorch_cpu

    # Run only the PyTorch CUDA version
    python3 main.py --platform pytorch_cuda

    # Run only the PyTorch MPS version
    python3 main.py --platform pytorch_mps

    # Run all available/supported versions sequentially
    python3 main.py --platform all

    # Run with different hyperparameters
    python3 main.py --platform all --epochs 10 --batch_size 256 --lr 0.005
    ```
    The script will automatically skip platforms if the required hardware/software (like CUDA or MPS) is not detected.
    
3.  **Analyze Results:**
    After running the experiments, analyze the logged metrics and generate plots:
    ```bash
    python3 evaluation/compare_results.py
    ```
    This will print a summary table to the console and save a comparison plot to `results/comparison_plot_combined.png`.

## Metrics Measured

* Training Time per Epoch (seconds)
* Total Training Time per platform (seconds)
* Training Throughput (samples/second) per epoch
* Training Loss & Accuracy per epoch
* Test Accuracy per epoch
* Process RAM Usage (MB) per epoch
* GPU VRAM Usage (MB) per epoch (for PyTorch CUDA)

## References

* Krizhevsky, Sutskever, & Hinton (2012). ImageNet Classification with Deep Convolutional Neural Networks.
* Sze, Chen, Yang & Emer (2017). Efficient Processing of Deep Neural Networks: A Tutorial and Survey.
* Hubner, Hu, Peng & Markidis (2025). Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency.
