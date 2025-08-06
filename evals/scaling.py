import json
import os
import torch
from model import DynamoModel  # Your DYNAMO model class
from train import train_model  # Your training function
from evaluate_dynamo import evaluate_model  # Your evaluation function
from torch.utils.data import DataLoader
from test_dataset import TestDataset  # Your dataset class

def run_scaling_experiments(config_path: str, data_path: str, device: str):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Iterate over each model size
    for model_size in config["model_sizes"]:
        print(f"Processing model size: {model_size}")
        transformer_name = config["transformer"][model_size]
        batch_size = config["batch_size"][model_size]
        learning_rate = config["learning_rate"][model_size]
        
        # Set up output directories
        results_dir = f"results/{model_size}"
        os.makedirs(os.path.join(results_dir, "weights"), exist_ok=True)
        weights_path = os.path.join(results_dir, "weights", "dynamo.pt")
        
        # Initialize the DYNAMO model
        model = DynamoModel(transformer_name, config).to(device)
        
        # Train the model (assumes train_model saves logs/weights as needed)
        train_model(model, data_path, batch_size, learning_rate, device, config["epochs"])
        
        # Save trained model weights
        torch.save(model.state_dict(), weights_path)
        
        # Evaluate the model (example for Batch 1; adjust as per your test setup)
        test_dataset = TestDataset(data_path, batch_idx=0)  # Batch 1
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        metrics = evaluate_model(model, test_loader, device, config)
        
        # Save evaluation metrics
        with open(os.path.join(results_dir, "eval_results.json"), 'w') as f:
            json.dump(metrics, f)
        
        print(f"Completed {model_size}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run scaling experiments for DYNAMO.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/dynamodata.json', help='Path to data')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda)')
    args = parser.parse_args()
    run_scaling_experiments(args.config, args.data_path, args.device)
