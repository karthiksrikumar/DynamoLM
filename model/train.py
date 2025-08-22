# train.py - Complete training script
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import logging
import time

# Local imports
from model import create_dynamo_model
# from processing import load_processed_data, create_dataloaders, validate_processed_data
# These functions don't exist in processing.py, commenting out for now
from utils import get_linear_scheduler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamoTrainer:
    def __init__(self, model, train_loader, test_loader, node_list, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.node_list = node_list
        self.config = config
        
        # Training parameters
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = config.get('learning_rate', 5e-5)
        self.epochs = config.get('epochs', 3)
        self.warmup_steps = config.get('warmup_steps', 100)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.save_steps = config.get('save_steps', 500)
        self.eval_steps = config.get('eval_steps', 250)
        self.output_dir = config.get('output_dir', 'checkpoints')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Calculate total steps
        self.total_steps = len(self.train_loader) * self.epochs
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.eval_losses = []
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            loss = self._forward_step(batch)
            
            # Backward pass
            self._backward_step(loss)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Evaluation and saving
            if self.global_step > 0 and self.global_step % self.eval_steps == 0:
                eval_loss = self.evaluate()
                self.eval_losses.append(eval_loss)
                logger.info(f"Step {self.global_step}: eval_loss = {eval_loss:.4f}")
                
                # Save if best model
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint(is_best=True)
                    logger.info(f"New best model saved with eval_loss = {eval_loss:.4f}")
                
                self.model.train()  # Back to training mode
            
            if self.global_step > 0 and self.global_step % self.save_steps == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        return total_loss / num_batches
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if key == 'edge_indices':
                # Handle list of edge indices
                device_batch[key] = [ei.to(self.device) for ei in value]
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _forward_step(self, batch: Dict) -> torch.Tensor:
        """Forward pass through the model"""
        # Handle edge indices properly - for now use first one but ensure consistency
        # TODO: Implement proper batched graph processing for multiple edge indices
        edge_indices = batch['edge_indices'][0] if batch['edge_indices'] and len(batch['edge_indices']) > 0 else None
        
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            time=batch['time'],
            edge_indices=edge_indices,
            labels=batch['labels']
        )
        
        return outputs.loss
    
    def _backward_step(self, loss: torch.Tensor):
        """Backward pass and optimization"""
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
    
    def evaluate(self) -> float:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss = self._forward_step(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self) -> Dict:
        """Full training loop"""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Train epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Evaluate
            eval_loss = self.evaluate()
            self.eval_losses.append(eval_loss)
            
            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Train loss: {train_loss:.4f}")
            logger.info(f"  Eval loss: {eval_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch=epoch)
            
            # Save best model
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.save_checkpoint(is_best=True)
                logger.info(f"New best model saved!")
        
        # Final save
        self.save_checkpoint(is_final=True)
        
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Return training stats
        return {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'best_loss': self.best_loss,
            'total_steps': self.global_step,
            'training_time': training_time
        }
    
    def save_checkpoint(self, epoch: int = None, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
            'node_list': self.node_list,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses
        }
        
        if is_best:
            save_path = os.path.join(self.output_dir, 'best_model.pt')
        elif is_final:
            save_path = os.path.join(self.output_dir, 'final_model.pt')
        elif epoch is not None:
            save_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        else:
            save_path = os.path.join(self.output_dir, f'checkpoint_step_{self.global_step}.pt')
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train DYNAMO QA model")
    
    # Data arguments
    parser.add_argument('--train_data', type=str, default='processed_data/train_data.pt',
                        help="Path to training data")
    parser.add_argument('--test_data', type=str, default='processed_data/test_data.pt',
                        help="Path to test data")
    
    # Model arguments
    parser.add_argument('--transformer', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Transformer model name")
    parser.add_argument('--config', type=str, default='model/configs.json',
                        help="Path to model config file")
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help="Output directory for checkpoints")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument('--eval_steps', type=int, default=250,
                        help="Evaluate every N steps")
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of data loading workers")
    parser.add_argument('--validate_only', action='store_true',
                        help="Only validate data without training")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Validate data files
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data not found: {args.test_data}")
    
    # Validate data
    print("Validating data...")
    if not validate_processed_data(args.train_data):
        raise ValueError("Training data validation failed")
    if not validate_processed_data(args.test_data):
        raise ValueError("Test data validation failed")
    
    if args.validate_only:
        print("Data validation passed!")
        return
    
    # Load data
    print("Loading datasets...")
    train_dataset, node_list = load_processed_data(args.train_data)
    test_dataset, _ = load_processed_data(args.test_data)
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model config
    if os.path.exists(args.config):
        model_config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Config file not found: {args.config}, using defaults")
        model_config = {}
    
    # Override config with command line arguments
    model_config.update({
        'transformer': args.transformer,
        'device': device,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'max_grad_norm': args.max_grad_norm,
        'output_dir': args.output_dir,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'batch_size': args.batch_size
    })
    
    # Create model
    print("Creating model...")
    model = create_dynamo_model(args.transformer, **model_config)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = DynamoTrainer(model, train_loader, test_loader, node_list, model_config)
    
    # Train model
    print("Starting training...")
    training_stats = trainer.train()
    
    # Print final results
    print("\n=== TRAINING COMPLETED ===")
    print(f"Best validation loss: {training_stats['best_loss']:.4f}")
    print(f"Total training steps: {training_stats['total_steps']}")
    print(f"Training time: {training_stats['training_time']:.2f} seconds")
    
    # Save training stats
    stats_path = os.path.join(args.output_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        # Convert any tensors to lists for JSON serialization
        stats_to_save = {
            'train_losses': training_stats['train_losses'],
            'eval_losses': training_stats['eval_losses'],
            'best_loss': training_stats['best_loss'],
            'total_steps': training_stats['total_steps'],
            'training_time': training_stats['training_time'],
            'config': model_config
        }
        json.dump(stats_to_save, f, indent=2)
    
    print(f"Training statistics saved to: {stats_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")
    print(f"Final model saved to: {os.path.join(args.output_dir, 'final_model.pt')}")

if __name__ == "__main__":
    main()
