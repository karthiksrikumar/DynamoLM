import json
import torch
from torch.utils.data import Dataset, DataLoader
from model import DynamoModel
from models.compRAG import RAGModel
from models.full_finetune import FullFinetuneModel
from metrics import compute_accuracy, compute_drift
import argparse
import os
from typing import Dict, List, Tuple, Optional

class TestDataset(Dataset):
    def __init__(self, processed_data: List[Dict], batch_idx: Optional[int] = None):
        """
        Initialize test dataset from processed data.
        
        Args:
            processed_data: List of processed data dictionaries
            batch_idx: Optional batch index for temporal filtering (0, 1, 2)
        """
        self.data = []
        
        if batch_idx is not None:
            # Filter by temporal batch for drift evaluation
            # Updated to match temporal_data_splitter.py ranges (2025-2026)
            from datetime import datetime
            batch_ranges = [
                (datetime(2025, 1, 1).timestamp(), datetime(2025, 8, 31).timestamp()),   # Batch 1: Jan-Aug 2025 (Training)
                (datetime(2025, 9, 1).timestamp(), datetime(2025, 12, 31).timestamp()),  # Batch 2: Sep-Dec 2025 (Validation)
                (datetime(2026, 1, 1).timestamp(), datetime(2026, 12, 31).timestamp())   # Batch 3: Jan-Dec 2026 (Test)
            ]
            start_time, end_time = batch_ranges[batch_idx]
            
            for item in processed_data:
                # Use timestamp field for consistency
                item_time = item.get('timestamp', item.get('time', 0))
                if isinstance(item_time, torch.Tensor):
                    item_time = item_time.item()
                
                if start_time <= item_time <= end_time:
                    self.data.append(item)
        else:
            # Use all data
            self.data = processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        item = self.data[idx]
        return (
            item['input_ids'],
            item['attention_mask'], 
            item['time'],
            item['labels'],  # Use labels as target for classification
            item['edge_index']
        )

def evaluate_model(model, dataloader, device: str, config: dict) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset with consistent input handling.

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
            input_ids, attention_mask, time, labels, edge_indices = batch
            
            # Move tensors to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            time = time.to(device)
            labels = labels.to(device)
            edge_indices = edge_indices.to(device)
            
            # Get model outputs - DYNAMO gets additional features as part of the research design
            # This is intentional and fair: we're testing whether temporal+graph features improve performance
            if isinstance(model, (RAGModel, FullFinetuneModel)):
                # Baseline models only get text inputs (their designed capability)
                logits = model(input_ids, attention_mask)
            else:
                # DYNAMO model gets temporal and graph features (its designed capability)
                logits = model(input_ids, attention_mask, time, [edge_indices])
            
            # For generative models, we need to extract predictions differently
            if hasattr(logits, 'logits'):
                logits = logits.logits
            
            # Get predictions - for QA, we compare generated text
            if logits.dim() == 3:  # [batch, seq_len, vocab_size]
                # For generative models, use the last token's prediction
                preds = torch.argmax(logits[:, -1, :], dim=-1)
            else:  # [batch, num_classes]
                # For classification models
                preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().tolist())
            
            # Extract labels - for QA, use the last non-padding token
            if labels.dim() == 2:  # [batch, seq_len]
                # Find last non-padding token for each sequence
                batch_labels = []
                for i in range(labels.size(0)):
                    seq_labels = labels[i]
                    # Find last non-padding token (assuming -100 is padding)
                    valid_indices = (seq_labels != -100).nonzero(as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        batch_labels.append(seq_labels[valid_indices[-1]].item())
                    else:
                        batch_labels.append(0)  # Fallback
                true_labels.extend(batch_labels)
            else:  # [batch]
                true_labels.extend(labels.cpu().tolist())
    
    accuracy = compute_accuracy(predictions, true_labels)
    return {'accuracy': accuracy}

def main(args):
    # Setup reproducibility
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
    try:
        from reproducibility import set_all_seeds, setup_deterministic_training
    except ImportError:
        print("Warning: Could not import reproducibility module, using basic seed setting")
        import random
        import numpy as np
        def set_all_seeds(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        def setup_deterministic_training():
            torch.backends.cudnn.deterministic = True
    
    set_all_seeds(42)
    setup_deterministic_training()
    
    device = args.device
    results = {}

    # Load configurations
    with open(args.config, 'r') as f:
        dynamo_config = json.load(f)
    
    # Process data using the same temporal splitting as training
    from model.fullpipeline import process_data
    print("Processing data with temporal splits...")
    train_data, val_data, test_data, node_list, tokenizer = process_data(
        args.data_path, dynamo_config.get('transformer', 'meta-llama/Llama-2-7b-hf'), 
        use_temporal_split=True
    )
    
    # Verify temporal consistency to prevent data leakage
    print(f"Data split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print("⚠️  IMPORTANT: Only using validation (Batch 1) and test (Batch 2) data for evaluation")
    print("   Training data (Batch 0) is excluded to prevent data leakage")
    
    # Initialize models
    dynamo = DynamoModel(dynamo_config).to(device)
    rag = RAGModel(dynamo_config['transformer']).to(device)
    full_ft = FullFinetuneModel(dynamo_config['transformer']).to(device)

    # Load pre-trained weights (assumed paths from training)
    dynamo.load_state_dict(torch.load(args.dynamo_weights, map_location=device))
    rag.load_state_dict(torch.load(args.rag_weights, map_location=device))
    full_ft.load_state_dict(torch.load(args.full_ft_weights, map_location=device))

    # Evaluate primarily on test data (Batch 2: 2026) 
    # Can also measure performance on validation data (Batch 1: Sep-Dec 2025) for drift analysis
    # Batch 0 (training: Jan-Aug 2025) is NEVER used for evaluation to prevent data leakage
    
    # For proper evaluation, we should mainly focus on test data (Batch 2)
    # But we can also check validation performance (Batch 1) to measure temporal drift
    batch_accuracies = {1: [], 2: []}  # 1=validation, 2=test
    
    # Primary evaluation on test set (Batch 2: 2026)
    print("Evaluating on test set (Jan-Dec 2026)...")
    test_dataset_batch = TestDataset(test_data, 2)  # Batch 2 = test data
    
    if len(test_dataset_batch) > 0:
        test_loader = DataLoader(test_dataset_batch, batch_size=dynamo_config['batch_size'], shuffle=False)
        
        # Evaluate each model on test set
        for model, name in [(dynamo, 'DYNAMO'), (rag, 'RAG'), (full_ft, 'Full FT')]:
            metrics = evaluate_model(model, test_loader, device, dynamo_config)
            batch_accuracies[2].append((name, metrics['accuracy']))
            print(f"  {name} test accuracy: {metrics['accuracy']:.2f}%")
    else:
        print("⚠️  No test data found for 2026 period")
    
    # Optional: Also evaluate on validation set for drift measurement
    print("\nEvaluating on validation set (Sep-Dec 2025) for drift analysis...")
    val_dataset_batch = TestDataset(val_data, 1)  # Use validation data directly
    
    if len(val_dataset_batch) > 0:
        val_loader = DataLoader(val_dataset_batch, batch_size=dynamo_config['batch_size'], shuffle=False)
        
        # Evaluate each model on validation set
        for model, name in [(dynamo, 'DYNAMO'), (rag, 'RAG'), (full_ft, 'Full FT')]:
            metrics = evaluate_model(model, val_loader, device, dynamo_config)
            batch_accuracies[1].append((name, metrics['accuracy']))
            print(f"  {name} validation accuracy: {metrics['accuracy']:.2f}%")
    else:
        print("⚠️  No validation data found for Sep-Dec 2025 period")
        
        # Fallback: If no validation data, we can't measure drift properly
        print("   Cannot measure temporal drift without validation data")
    
    # Compute average accuracy and drift (only using validation and test batches)
    for name in ['DYNAMO', 'RAG', 'Full FT']:
        accuracies = [acc for b in batch_accuracies.values() for n, acc in b if n == name]
        if len(accuracies) > 0:
            avg_accuracy = sum(accuracies) / len(accuracies)
            drift = compute_drift(accuracies) if len(accuracies) > 1 else 0.0
            results[name] = {'accuracy': avg_accuracy, 'drift': drift}
        else:
            print(f"⚠️  No evaluation data found for {name}")
            results[name] = {'accuracy': 0.0, 'drift': 0.0}
    
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
