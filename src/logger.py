# src/logger.py
import csv
import os
import time
import platform
import psutil # For basic memory usage (pip install psutil)
import torch # Import torch to check for cuda

class PerformanceLogger:
    """Logs performance metrics during training to platform-specific CSV files."""

    def __init__(self, log_directory='results'):
        """
        Initializes the logger. Ensures the log directory exists.

        Args:
            log_directory (str): Path to the directory where log files will be saved.
        """
        self.log_directory = log_directory
        self.metrics_list = [] # Store metrics temporarily if needed
        self._start_time = None
        self._epoch_start_time = None
        self._platform_name = "Unknown"
        self._device = None
        self._current_filepath = None # Path to the log file for the current experiment

        # Create results directory if it doesn't exist
        os.makedirs(self.log_directory, exist_ok=True)

        # Define fieldnames
        self._fieldnames = [
            'Platform', 'Epoch', 'Epoch Time (s)', 'Train Loss',
            'Train Accuracy', 'Test Accuracy', 'Throughput (samples/s)',
            'Timestamp', 'System RAM Used (MB)', 'Process RAM Used (MB)',
            'GPU VRAM Used (MB)' 
        ]

        # Get system info once
        self.system_info = f"OS: {platform.system()} {platform.release()}, Arch: {platform.machine()}"
        print(f"Logger initialized. Logging to directory: {self.log_directory}")
        print(f"System: {self.system_info}")


    def _write_header_if_needed(self, filepath):
        """Writes the CSV header row if the specified file is new or empty."""
        # Check if file needs header right before writing
        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        if write_header:
            try:
                # Open in 'w' mode only if header needs writing (creates/truncates file)
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self._fieldnames)
                # print(f"CSV header written to {filepath}.") # Optional: less verbose
            except IOError as e:
                print(f"Error writing CSV header to {filepath}: {e}")


    def start_experiment(self, platform_name: str, device: str = None):
        """Call this at the beginning of an experiment run."""
        self._platform_name = platform_name
        self._device = device # Store device type (e.g., 'cuda', 'mps')
        # Construct the specific file path for this platform run
        self._current_filepath = os.path.join(self.log_directory, f"metrics_{self._platform_name}.csv")

        # Write header if this specific file needs it (will create/overwrite if writing header)
        self._write_header_if_needed(self._current_filepath)

        self._start_time = time.time()
        print(f"\n--- Starting Experiment: {self._platform_name} (Device: {self._device or 'N/A'}) ---")
        print(f"Logging metrics to: {self._current_filepath}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")


    def start_epoch(self):
        """Call this at the beginning of each epoch."""
        self._epoch_start_time = time.time()


    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  test_acc: float, num_train_samples: int):
        """
        Logs metrics for a completed epoch and appends to the platform-specific CSV file.

        Args:
            epoch (int): The current epoch number (1-based).
            train_loss (float): Average training loss for the epoch.
            train_acc (float): Training accuracy for the epoch.
            test_acc (float): Test accuracy for the epoch.
            num_train_samples (int): Number of samples processed during training in this epoch.
        """
        if self._current_filepath is None:
            print("Error: Logger experiment not started (call start_experiment()). Cannot log.")
            return

        if self._epoch_start_time is None:
            print("Warning: Epoch timer not started (call start_epoch()). Cannot log epoch time/throughput.")
            epoch_time = -1.0
            throughput = -1.0
        else:
            epoch_time = time.time() - self._epoch_start_time
            throughput = num_train_samples / epoch_time if epoch_time > 0 else 0.0

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # Get memory usage (basic system and process RAM)
        try:
            system_mem = psutil.virtual_memory()
            process_mem = psutil.Process(os.getpid()).memory_info()
            system_ram_used_mb = (system_mem.total - system_mem.available) / (1024 * 1024)
            process_ram_used_mb = process_mem.rss / (1024 * 1024) # Resident Set Size
        except Exception as e:
            print(f"Warning: Could not get memory usage using psutil: {e}")
            system_ram_used_mb = -1.0
            process_ram_used_mb = -1.0

        # --- VRAM Logging ---
        gpu_vram_used_mb = -1.0 # Default value if not applicable or error
        try:
            if self._device == 'cuda' and torch.cuda.is_available():
                 gpu_vram_used_mb = torch.cuda.memory_allocated(self._device) / (1024 * 1024)
            elif self._device == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                 pass # Keep default -1 for MPS VRAM
        except Exception as e:
            print(f"Warning: Could not get GPU VRAM usage for device {self._device}: {e}")


        log_entry = {
            'Platform': self._platform_name,
            'Epoch': epoch,
            'Epoch Time (s)': round(epoch_time, 3),
            'Train Loss': round(train_loss, 5),
            'Train Accuracy': round(train_acc, 5),
            'Test Accuracy': round(test_acc, 5),
            'Throughput (samples/s)': round(throughput, 2),
            'Timestamp': timestamp,
            'System RAM Used (MB)': round(system_ram_used_mb, 2),
            'Process RAM Used (MB)': round(process_ram_used_mb, 2),
            'GPU VRAM Used (MB)': round(gpu_vram_used_mb, 2)
        }
        self.metrics_list.append(log_entry) # Optional: keep in memory

        # Append to the specific CSV file for this platform run
        try:
            # Header should already exist from start_experiment
            # Open in append mode ('a')
            with open(self._current_filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                # Check if file is empty after opening (edge case: file deleted between start and log)
                f.seek(0, os.SEEK_END)
                is_empty = f.tell() == 0
                if is_empty: # If file somehow became empty, rewrite header
                   self._write_header_if_needed(self._current_filepath) # This reopens in 'w'
                   # Need to reopen in 'a' again to append
                   with open(self._current_filepath, 'a', newline='') as fa:
                       writer_a = csv.DictWriter(fa, fieldnames=self._fieldnames)
                       writer_a.writerow(log_entry)
                else:
                    writer.writerow(log_entry) # Append normally
        except IOError as e:
            print(f"Error writing epoch log to CSV {self._current_filepath}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during logging: {e}")


        # Print summary to console
        vram_str = f" | VRAM: {gpu_vram_used_mb:.1f}MB" if gpu_vram_used_mb >= 0 else ""
        print(f"Epoch {epoch:02d} [{self._platform_name}] | "
              f"Time: {epoch_time:.2f}s | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | Throughput: {throughput:.0f} samples/s | "
              f"RAM: {process_ram_used_mb:.1f}MB{vram_str}")


    def end_experiment(self):
        """Call this at the end of an experiment run."""
        if self._start_time is None:
            print("Warning: Experiment timer not started. Cannot log total time.")
            return
        total_time = time.time() - self._start_time
        print(f"--- Experiment End: {self._platform_name} ---")
        print(f"Total Training Time: {total_time:.2f}s")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Reset timers and current file path for next potential experiment
        self._start_time = None
        self._epoch_start_time = None
        self._device = None
        self._current_filepath = None


if __name__ == '__main__':
    print("--- Testing Logger (Separate Files) ---")
    test_log_dir = 'results_test'
    test_logger = PerformanceLogger(log_directory=test_log_dir)

    # Test CPU logging
    platform_cpu = "TestPlatform_CPU"
    test_logger.start_experiment(platform_cpu, device='cpu')
    num_samples = 10000
    test_logger.start_epoch()
    time.sleep(0.1) # Simulate work
    test_logger.log_epoch(epoch=1, train_loss=0.5, train_acc=0.8, test_acc=0.75, num_train_samples=num_samples)
    test_logger.log_epoch(epoch=2, train_loss=0.4, train_acc=0.85, test_acc=0.80, num_train_samples=num_samples)
    test_logger.end_experiment()
    cpu_log_file = os.path.join(test_log_dir, f"metrics_{platform_cpu}.csv")
    print(f"Check CPU log file: {cpu_log_file}")
    assert os.path.exists(cpu_log_file)

    # Test CUDA logging (mock VRAM)
    platform_cuda = "TestPlatform_CUDA"
    if torch.cuda.is_available():
         test_logger.start_experiment(platform_cuda, device='cuda')
         test_logger.start_epoch()
         time.sleep(0.1)
         _ = torch.randn(100, 100, device='cuda') # Simulate allocating memory
         test_logger.log_epoch(epoch=1, train_loss=0.4, train_acc=0.85, test_acc=0.80, num_train_samples=num_samples)
         test_logger.end_experiment()
         del _
         torch.cuda.empty_cache()
         cuda_log_file = os.path.join(test_log_dir, f"metrics_{platform_cuda}.csv")
         print(f"Check CUDA log file: {cuda_log_file}")
         assert os.path.exists(cuda_log_file)

    else:
        print("Skipping CUDA logger test as CUDA is not available.")

    # Clean up test directory
    # import shutil
    # if os.path.exists(test_log_dir):
    #     shutil.rmtree(test_log_dir)
    #     print(f"Removed test directory: {test_log_dir}")

