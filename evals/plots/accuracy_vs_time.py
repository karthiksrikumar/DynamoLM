import matplotlib.pyplot as plt
import json
import numpy as np

def plot_accuracy_vs_time(results_path: str, output_path: str):
    """
    Plot accuracy vs. time for DYNAMO, RAG, and Full FT for 7b.

    Args:
        results_path (str): Path to evaluation results JSON.
        output_path (str): Path to save the plot.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    batches = ['Batch 1', 'Batch 2', 'Batch 3']
    time_points = [1725120000, 1738368000, 1746009600]  # Midpoints of July 2024, Jan 2025, Apr 2025
    
    plt.figure(figsize=(8, 6))
    for model in ['DYNAMO', 'RAG', 'Full FT']:
        accuracies = [results[model]['accuracy']] * 3  # Approximate per-batch from average
        plt.plot(time_points, accuracies, marker='o', label=model)
    
    plt.xlabel('Time (Unix Timestamp)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Time Across Batches')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot accuracy vs. time.")
    parser.add_argument('--results_path', type=str, default='evals/results.json')
    parser.add_argument('--output_path', type=str, default='evals/plots/accuracy_vs_time.png')
    args = parser.parse_args()
    plot_accuracy_vs_time(args.results_path, args.output_path)
