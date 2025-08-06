import json
import matplotlib.pyplot as plt
import os

def plot_accuracy_vs_model_size(results_dir: str, output_path: str):
    model_sizes = ['llama-7b', 'llama-13b', 'llama-30b']
    accuracies = []
    
    # Load accuracy for each model size
    for size in model_sizes:
        result_file = os.path.join(results_dir, size, 'eval_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)
                accuracies.append(results['accuracy'])  # Adjust key based on your metrics
        else:
            print(f"Warning: Results for {size} not found.")
            accuracies.append(None)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(model_sizes, accuracies, marker='o', linestyle='-')
    plt.xlabel('Model Size')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Model Size for DYNAMO')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    os.makedirs('evals/plots', exist_ok=True)
    plot_accuracy_vs_model_size('results', 'evals/plots/accuracy_vs_model_size.png')
