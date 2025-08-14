import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from model import DynamoModel
import os
from typing import List, Tuple

class DynamoDataset(Dataset):
    """Custom dataset for DYNAMO, loading text, time, and graph data from JSON."""
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to JSON file containing training data.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found.")
        
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        self.data = []
        for item in raw_data:
            required_fields = ['input_ids', 'attention_mask', 'time', 'label', 'edge_indices']
            if not all(field in item for field in required_fields):
                raise ValueError(f"Missing required fields in data item: {item}")
            
            input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
            time = torch.tensor(item['time'], dtype=torch.float)
            label = torch.tensor(item['label'], dtype=torch.long)
            edge_indices = torch.tensor(item['edge_indices'], dtype=torch.long)
            
            self.data.append((input_ids, attention_mask, time, label, edge_indices))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.data[idx]

def compute_causal_loss(model: nn.Module, batch: Tuple, config: dict) -> torch.Tensor:
    """
    Compute graph-based causal regularization loss.

    For each causal edge in the batch, compare model predictions for the effect node
    under the same intervention at different times, using KL divergence as a proxy.

    Args:
        model (nn.Module): DYNAMO model instance.
        batch (tuple): Batch of (input_ids, attention_mask, time, labels, edge_indices).
        config (dict): Configuration dictionary.

    Returns:
        torch.Tensor: Causal regularization loss.
    """
    input_ids, attention_mask, time, _, edge_indices = batch
    batch_size = input_ids.size(0)
    device = input_ids.device

    if not config.get('use_gnn', True):
        return torch.tensor(0.0, device=device)

    causal_loss = 0.0
    num_pairs = 0

    # For each sample in the batch
    for i in range(batch_size):
        t_i = time[i]
        edges_i = edge_indices[i]  # [2, num_edges]
        if edges_i.size(1) == 0:  # Skip if no edges
            continue

        # Simulate intervention: perturb input by focusing on effect node
        for edge in edges_i.t():  # Iterate over edges
            # Create a synthetic time point (e.g., t_i + delta)
            t_j = t_i + torch.tensor(config.get('time_delta', 86400.0), device=device)  # 1 day shift
            logits_i = model(input_ids[i:i+1], attention_mask[i:i+1], t_i.unsqueeze(0), edges_i)
            logits_j = model(input_ids[i:i+1], attention_mask[i:i+1], t_j.unsqueeze(0), edges_i)

            # Compute KL divergence between softmax outputs
            probs_i = F.softmax(logits_i, dim=-1)
            probs_j = F.softmax(logits_j, dim=-1)
            kl_div = F.kl_div(probs_i.log(), probs_j, reduction='batchmean')
            causal_loss += kl_div
            num_pairs += 1

    return causal_loss / max(1, num_pairs) if num_pairs > 0 else torch.tensor(0.0, device=device)

def train_model(config: dict, data_path: str = 'data/dynamodata.json', device: str = 'cuda'):
    """
    Train the DYNAMO model with the given configuration and data.

    Args:
        config (dict): Configuration dictionary with model hyperparameters.
        data_path (str): Path to JSON data file.
        device (str): Device to train on (e.g., 'cuda' or 'cpu').
    """
    # Initialize dataset and dataloader
    try:
        train_dataset = DynamoDataset(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {data_path}: {str(e)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Model and optimizer
    model = DynamoModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, time, labels, edge_indices = [x.to(device) for x in batch]
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask, time, edge_indices)
            loss = criterion(logits, labels)
            
            # Causal regularization
            if config.get('use_causal_reg', False):
                causal_loss = compute_causal_loss(model, batch, config)
                loss += config['lambda_causal'] * causal_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {total_loss / len(train_loader):.4f}")

    return model

if __name__ == "__main__":
    # Parse command-line arguments for variant selection
    parser = argparse.ArgumentParser(description="Train DYNAMO model with specified variant.")
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'no_time2vec', 'no_gnn', 'no_causal_reg'],
                        help="Model variant to train")
    parser.add_argument('--data_path', type)
