import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from model import DynamoModel, create_dynamo_model
import os
from typing import List, Tuple, Dict, Any
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

class DynamoQADataset(Dataset):
    """Custom dataset for DYNAMO QA, loading from preprocessed .pt file."""
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to .pt file containing preprocessed training data.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found.")
        
        # Load the preprocessed data
        data_dict = torch.load(data_path)
        self.processed_data = data_dict["processed_data"]
        self.node_list = data_dict["node_list"]

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.processed_data[idx]
    
    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-length edge indices.
        """
        # Stack tensors that have consistent shapes
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        time = torch.stack([item['time'] for item in batch])
        
        # Edge indices need special handling as they may have different shapes
        edge_indices = [item['edge_index'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'time': time,
            'edge_indices': edge_indices
        }

def train_model(config: dict, data_path: str = 'processed_dynamodata_qa.pt', device: str = 'cuda'):
    """
    Train the DYNAMO model with the given configuration and data.

    Args:
        config (dict): Configuration dictionary with model hyperparameters.
        data_path (str): Path to preprocessed data file.
        device (str): Device to train on (e.g., 'cuda' or 'cpu').
    """
    # Initialize dataset and dataloader
    try:
        train_dataset = DynamoQADataset(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {data_path}: {str(e)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    # Model and optimizer
    model = create_dynamo_model(config['transformer_path'], **config)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    num_epochs = config['epochs']
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(config.get('warmup_ratio', 0.1) * num_training_steps)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            time = batch['time'].to(device)
            edge_indices = batch['edge_indices']
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                time=time,
                edge_indices=edge_indices,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.get('max_grad_norm', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['max_grad_norm']
                )
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % config.get('log_interval', 10) == 0:
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 1) == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), "dynamo_qa_model_final.pth")
    print("Training completed. Final model saved to dynamo_qa_model_final.pth")
    
    return model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train DYNAMO QA model.")
    parser.add_argument('--data_path', type=str, default='processed_dynamodata_qa.pt',
                        help="Path to preprocessed data file")
    parser.add_argument('--transformer', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="HuggingFace transformer model name")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Training batch size")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help="Ratio of training steps for warmup")
    parser.add_argument('--log_interval', type=int, default=10,
                        help="Number of steps between logging")
    parser.add_argument('--save_interval', type=int, default=1,
                        help="Number of epochs between checkpoints")
    parser.add_argument('--use_time2vec', type=bool, default=True,
                        help="Whether to use Time2Vec embeddings")
    parser.add_argument('--use_gnn', type=bool, default=True,
                        help="Whether to use GNN component")
    parser.add_argument('--time2vec_dim', type=int, default=128,
                        help="Dimension of Time2Vec embeddings")
    parser.add_argument('--node_dim', type=int, default=256,
                        help="Dimension of node embeddings")
    parser.add_argument('--gnn_output_dim', type=int, default=256,
                        help="Output dimension of GNN")
    parser.add_argument('--gnn_layers', type=int, default=2,
                        help="Number of GNN layers")
    parser.add_argument('--fused_dim', type=int, default=512,
                        help="Dimension of fused features")
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument('--pooling_type', type=str, default='mean',
                        choices=['mean', 'max', 'sum', 'attention'],
                        help="Graph pooling type")
    parser.add_argument('--freeze_transformer', type=bool, default=False,
                        help="Whether to freeze transformer parameters")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to train on")
    
    args = parser.parse_args()
    
    # Create config dictionary
    config = {
        'transformer_path': args.transformer,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'warmup_ratio': args.warmup_ratio,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'use_time2vec': args.use_time2vec,
        'use_gnn': args.use_gnn,
        'time2vec_dim': args.time2vec_dim,
        'node_dim': args.node_dim,
        'gnn_output_dim': args.gnn_output_dim,
        'gnn_layers': args.gnn_layers,
        'fused_dim': args.fused_dim,
        'dropout_rate': args.dropout_rate,
        'pooling_type': args.pooling_type,
        'freeze_transformer': args.freeze_transformer,
    }
    
    # Train the model
    model = train_model(config, args.data_path, args.device)

if __name__ == "__main__":
    main()
