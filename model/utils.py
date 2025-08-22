# utils.py - Fixed utility functions
import torch
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def get_linear_scheduler(optimizer, num_training_steps: int, warmup_steps: int):
    """Get linear scheduler with warmup"""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

def count_parameters(model) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_training_plot(train_losses: List[float], eval_losses: List[float], 
                      save_path: str = "training_plot.png"):
    """Save training loss plot"""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_losses, label='Eval Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plot saved to {save_path}")

def load_json_config(config_path: str) -> Dict:
    """Load JSON configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def save_json_config(config: Dict, save_path: str):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {save_path}")

def format_time(seconds: float) -> str:
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"

def get_device(device_str: str = 'auto') -> torch.device:
    """Get torch device"""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    return device

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
    else:
        print("CUDA not available")

def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate exact match accuracy"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.strip().lower() == target.strip().lower():
            correct += 1
    
    return correct / len(predictions)

def calculate_bleu_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate BLEU score (simplified version)"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothie = SmoothingFunction().method4
        scores = []
        
        for pred, target in zip(predictions, targets):
            pred_tokens = pred.strip().split()
            target_tokens = target.strip().split()
            
            if len(pred_tokens) == 0 or len(target_tokens) == 0:
                scores.append(0.0)
            else:
                score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothie)
                scores.append(score)
        
        return np.mean(scores)
        
    except ImportError:
        print("NLTK not available for BLEU score calculation")
        return 0.0

def evaluate_predictions(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Evaluate predictions with multiple metrics"""
    results = {
        'accuracy': calculate_accuracy(predictions, targets),
        'bleu_score': calculate_bleu_score(predictions, targets)
    }
    
    return results

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

def create_output_directory(base_dir: str, experiment_name: str = None) -> str:
    """Create output directory with timestamp"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"
    
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    import logging
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # Setup file handler if log_file provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def cleanup_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """Clean up old checkpoints, keeping only the most recent ones"""
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_step_') and file.endswith('.pt'):
            step_num = int(file.split('_')[-1].split('.')[0])
            checkpoint_files.append((step_num, os.path.join(checkpoint_dir, file)))
    
    # Sort by step number
    checkpoint_files.sort(key=lambda x: x[0])
    
    # Remove old checkpoints
    if len(checkpoint_files) > keep_last:
        for _, file_path in checkpoint_files[:-keep_last]:
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test GPU memory
    print_gpu_memory()
    
    # Test number formatting
    print(f"Formatted number: {format_number(1234567)}")
    
    # Test time formatting
    print(f"Formatted time: {format_time(3661)}")
    
    print("All tests passed!")
