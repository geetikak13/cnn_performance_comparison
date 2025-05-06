# evaluation/compare_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Define constants for file paths
RESULTS_DIR = 'results' # Directory where individual CSVs are stored
# OUTPUT_PLOT defines where the combined plot is saved
OUTPUT_PLOT = os.path.join(RESULTS_DIR, 'comparison_plot_combined.png')

def analyze_and_visualize(results_dir=RESULTS_DIR, output_plot=OUTPUT_PLOT):
    """
    Loads metrics from all individual platform CSV files in the results directory,
    combines them, prints a summary table, and generates comparison plots.

    Args:
        results_dir (str): Path to the directory containing metrics_*.csv files.
        output_plot (str): Path where the output plot PNG will be saved.
    """
    print(f"\n--- Analyzing Combined Results from '{results_dir}' ---")

    # Ensure the results directory exists (for saving plots)
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    # --- Find and Load Data ---
    # Use glob to find all files matching the pattern
    csv_pattern = os.path.join(results_dir, "metrics_*.csv")
    all_csv_files = glob.glob(csv_pattern)

    if not all_csv_files:
        print(f"Error: No metrics files found matching '{csv_pattern}'.")
        print("Please run the main training script first (e.g., 'python main.py --platform all').")
        return

    print(f"Found {len(all_csv_files)} metrics files:")
    # Read each CSV and store in a list of DataFrames
    all_dataframes = []
    for f in all_csv_files:
        try:
            df_single = pd.read_csv(f)
            print(f"  - Loaded '{os.path.basename(f)}' ({len(df_single)} records)")
            if not df_single.empty:
                all_dataframes.append(df_single)
            else:
                print(f"    Warning: File '{os.path.basename(f)}' is empty. Skipping.")
        except pd.errors.EmptyDataError:
             print(f"    Warning: File '{os.path.basename(f)}' is empty. Skipping.")
        except Exception as e:
            print(f"    Error reading file '{os.path.basename(f)}': {e}. Skipping.")

    if not all_dataframes:
        print("Error: No valid data loaded from any metrics files.")
        return

    # Concatenate all DataFrames into a single one
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nCombined data contains {len(df_combined)} total records from {len(all_dataframes)} files.")

    # Use the combined DataFrame 'df_combined' for subsequent analysis
    df = df_combined # Rename for consistency with the rest of the script

    # --- Data Cleaning/Preparation ---
    # Define expected numeric columns
    numeric_cols = ['Epoch Time (s)', 'Train Loss', 'Train Accuracy',
                    'Test Accuracy', 'Throughput (samples/s)',
                    'System RAM Used (MB)', 'Process RAM Used (MB)',
                    'GPU VRAM Used (MB)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Convert non-numeric to NaN
        else:
            print(f"Warning: Column '{col}' not found in combined data. Skipping numeric conversion.")


    # --- Summary Statistics ---
    print("\n--- Performance Summary (Averages & Last Epoch from Combined Data) ---")

    # Define aggregation dictionary, including VRAM if present
    agg_dict = {
        'avg_epoch_time': ('Epoch Time (s)', 'mean'),
        'total_training_time': ('Epoch Time (s)', 'sum'),
        'avg_throughput': ('Throughput (samples/s)', 'mean'),
        'last_epoch': ('Epoch', 'max'),
        'final_train_accuracy': ('Train Accuracy', 'last'),
        'final_test_accuracy': ('Test Accuracy', 'last'),
        'max_test_accuracy': ('Test Accuracy', 'max'),
        'avg_process_ram': ('Process RAM Used (MB)', 'mean')
    }
    # Add VRAM aggregation only if the column exists and has valid data
    if 'GPU VRAM Used (MB)' in df.columns and df['GPU VRAM Used (MB)'].notna().any():
        agg_dict['max_gpu_vram'] = ('GPU VRAM Used (MB)', 'max')

    # Group by platform and calculate summary metrics from the combined data
    summary = df.groupby('Platform').agg(**agg_dict).reset_index()


    # Format the summary table for better readability
    summary['avg_epoch_time'] = summary['avg_epoch_time'].map('{:.2f}s'.format)
    summary['total_training_time'] = summary['total_training_time'].map('{:.2f}s'.format)
    summary['avg_throughput'] = summary['avg_throughput'].map('{:.0f}'.format)
    summary['final_train_accuracy'] = summary['final_train_accuracy'].map('{:.2%}'.format)
    summary['final_test_accuracy'] = summary['final_test_accuracy'].map('{:.2%}'.format)
    summary['max_test_accuracy'] = summary['max_test_accuracy'].map('{:.2%}'.format)
    summary['avg_process_ram'] = summary['avg_process_ram'].map('{:.1f}MB'.format)
    if 'max_gpu_vram' in summary.columns:
        summary['max_gpu_vram'] = summary['max_gpu_vram'].apply(lambda x: f"{x:.1f}MB" if pd.notna(x) and x >= 0 else "N/A")


    # Print the formatted summary table
    try:
        # Reorder columns for better presentation
        cols_order = ['Platform', 'avg_epoch_time', 'total_training_time', 'avg_throughput',
                      'final_test_accuracy', 'max_test_accuracy', 'avg_process_ram']
        if 'max_gpu_vram' in summary.columns:
            cols_order.append('max_gpu_vram')
        cols_order.append('last_epoch') # Put last epoch at the end

        print(summary[cols_order].to_string(index=False))
    except Exception as e:
        print(f"Could not print summary table: {e}")

    print("\n--- Generating Comparison Plots from Combined Data ---")

    # --- Visualizations ---
    try:
        # Determine number of unique platforms for color mapping if needed
        n_platforms = df['Platform'].nunique()
        # Use a perceptually uniform palette like 'viridis'
        palette = sns.color_palette("viridis", n_platforms)

        plt.style.use('seaborn-v0_8-darkgrid') # Use a visually appealing style
        # Create a figure with multiple subplots (axes)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # Adjust figsize as needed
        fig.suptitle('CNN Performance Comparison Across Platforms (Combined Data)', fontsize=18, y=1.02)

        # Flatten axes array for easier iteration if more plots are added
        ax_flat = axes.flatten()
        plot_index = 0

        # Plot 1: Epoch Time vs. Epoch (Line Plot)
        ax = ax_flat[plot_index]
        sns.lineplot(data=df, x='Epoch', y='Epoch Time (s)', hue='Platform', marker='o', ax=ax, palette=palette)
        ax.set_title('Time per Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Epoch')
        ax.grid(True, linestyle='--', alpha=0.7)
        # Place legend outside the plot area
        legend = ax.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plot_index += 1

        # Plot 2: Throughput vs. Epoch (Line Plot)
        ax = ax_flat[plot_index]
        sns.lineplot(data=df, x='Epoch', y='Throughput (samples/s)', hue='Platform', marker='o', ax=ax, palette=palette)
        ax.set_title('Training Throughput per Epoch')
        ax.set_ylabel('Throughput (samples/second)')
        ax.set_xlabel('Epoch')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.get_legend().remove() # Remove redundant legend
        plot_index += 1

        # Plot 3: Test Accuracy vs. Epoch (Line Plot)
        ax = ax_flat[plot_index]
        sns.lineplot(data=df, x='Epoch', y='Test Accuracy', hue='Platform', marker='o', ax=ax, palette=palette)
        ax.set_title('Test Accuracy per Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.grid(True, linestyle='--', alpha=0.7)
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))
        ax.get_legend().remove() # Remove redundant legend
        plot_index += 1

        # Plot 4: Bar chart for Overall Average Throughput (using summary data)
        ax = ax_flat[plot_index]
        # Convert avg_throughput back to numeric for plotting
        summary_plot = summary.copy()
        try:
             summary_plot['avg_throughput_val'] = summary_plot['avg_throughput'].str.replace(r'[^\d.]', '', regex=True).astype(float)
        except AttributeError: # Handle case where it might already be float/int
             summary_plot['avg_throughput_val'] = pd.to_numeric(summary_plot['avg_throughput'], errors='coerce')


        sns.barplot(data=summary_plot.sort_values('avg_throughput_val', ascending=False),
                    x='Platform', y='avg_throughput_val', ax=ax, palette=palette)
        ax.set_title('Average Training Throughput (Overall)')
        ax.set_ylabel('Average Samples / Second')
        ax.set_xlabel('Platform')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=15)
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f')
        plot_index += 1

        # Adjust layout to prevent overlapping titles/labels and make space for legend
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.95]) # Adjust right margin for legend

        # Save the combined plot to a file
        plt.savefig(output_plot, bbox_inches='tight')
        print(f"Combined comparison plot saved successfully to '{output_plot}'")
        # plt.show() # Optionally display the plot interactively

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_and_visualize()
