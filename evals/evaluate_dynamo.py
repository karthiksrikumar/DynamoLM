import json
import torch
from torch.utils.data import Dataset, DataLoader
from model import DynamoModel
from baseline_models.rag import RAGModel
from baseline_models.full_finetune import FullFinetuneModel
from metrics import compute_accuracy, compute_drift
import argparse
import os
from typing import Dict, List, Tuple

class TestDataset(Dataset):
    def __init__(self, data_path: str, batch_idx: int):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found.")
        
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Filter for test set of the specified batch
        batch_ranges = [
            (1719792000, 1735689600),  # Batch 1: July–Dec 2024
            (1735689601, 1743350400),  # Batch 2: Jan–Mar 2025
            (1743350401, 1751328000)   # Batch 3: Apr–Jul 2025
        ]
        start_time, end_time = batch_ranges[batch_idx]
        self.data = []
        for item in raw_data:
            if 'is_test' in item and item['is_test'] and start_time <= item['time'] <= end_time:
                self.data.append((
                    torch.tensor(item['input_ids'], dtype=torch.long),
                    torch.tensor(item['attention_mask'], dtype=torch.long),
                    torch.tensor(item['time'], dtype=torch.float),
                    torch.tensor(item['label'], dtype=torch.long),
                    torch.tensor(item['edge_indices'], dtype=torch.long)
                ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.data[idx]

def evaluate_model(model, dataloader, device: str, config: dict) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset.

    Args:
        model: Model instance (DynamoModel, RAGModel, or FullFinetuneModel).
        dataloader: DataLoader for test data.
        device (str): Device to evaluate on.
        config (dict): Configuration dictionary.

    Returns:
        Dict[str, float]: Metrics (accuracy).
    """
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, time, labels, edge_indices = [x.to(device) for x in batch]
            if isinstance(model, (RAGModel, FullFinetuneModel)):
                logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask, time, edge_indices)
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    accuracy = compute_accuracy(predictions, true_labels)
    return {'accuracy': accuracy}

def main(args):
    device = args.device
    results = {}

    # Load configurations
    with open(args.config, 'r') as f:
        dynamo_config = json.load(f)
    
    # Initialize models
    dynamo = DynamoModel(dynamo_config).to(device)
    rag = RAGModel(dynamo_config['transformer']).to(device)
    full_ft = FullFinetuneModel(dynamo_config['transformer']).to(device)

    # Load pre-trained weights (assumed paths from training)
    dynamo.load_state_dict(torch.load(args.dynamo_weights, map_location=device))
    rag.load_state_dict(torch.load(args.rag_weights, map_location=device))
    full_ft.load_state_dict(torch.load(args.full_ft_weights, map_location=device))

    # Evaluate across batches
    batch_accuracies = {0: [], 1: [], 2: []}
    for batch_idx in range(3):  # Batch 1 → Batch 2, Batch 2 → Batch 3, Batch 3 → held-out
        test_loader = DataLoader(TestDataset(args.data_path, batch_idx), batch_size=dynamo_config['batch_size'])
        
        # Evaluate each model
        for model, name in [(dynamo, 'DYNAMO'), (rag, 'RAG'), (full_ft, 'Full FT')]:
            metrics = evaluate_model(model, test_loader, device, dynamo_config)
            batch_accuracies[batch_idx].append((name, metrics['accuracy']))
    
    # Compute average accuracy and drift
    for name in ['DYNAMO', 'RAG', 'Full FT']:
        accuracies = [acc for b in batch_accuracies.values() for n, acc in b if n == name]
        avg_accuracy = sum(accuracies) / len(accuracies)
        drift = compute_drift(accuracies)
        results[name] = {'accuracy': avg_accuracy, 'drift': drift}
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: Accuracy = {metrics['accuracy']:.1f}%, Drift = {metrics['drift']:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DYNAMO and baselines on Test 2.")
    parser.add_argument('--data_path', type=str, default='data/dynamodata.json', help="Path to test data")
    parser.add_argument('--config', type=str, default='config.json', help="Path to DYNAMO config")
    parser.add_argument('--dynamo_weights', type=str, default='weights/dynamo.pt', help="DYNAMO weights")
    parser.add_argument('--rag_weights', type=str, default='weights/rag.pt', help="RAG weights")
    parser.add_argument('--full_ft_weights', type=str, default='weights/full_ft.pt', help="Full FT weights")
    parser.add_argument('--output_path', type=str, default='evals/results.json', help="Path to save results")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda or cpu)")
    args = parser.parse_args()
    main(args)
