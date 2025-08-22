#!/usr/bin/env python3
"""
COMPLETE DYNAMO PIPELINE - Single File Solution
==================================================

Input: dynamodata.json
Output: Trained model ready for inference

Usage:
    python complete_pipeline.py                    # Full pipeline
    python complete_pipeline.py --quick_test       # Quick test (5 min)
    python complete_pipeline.py --data custom.json # Custom data file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, LlamaForCausalLM, 
    get_linear_schedule_with_warmup
)
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

import json
import os
import re
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings("ignore")

# Import reproducibility controls
try:
    from model.reproducibility import set_all_seeds, setup_deterministic_training, ReproducibleTrainer
except ImportError:
    try:
        from reproducibility import set_all_seeds, setup_deterministic_training, ReproducibleTrainer
    except ImportError:
        print("Warning: Reproducibility module not found, using basic seed setting")
        import random
        import numpy as np
        def set_all_seeds(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        def setup_deterministic_training():
            torch.backends.cudnn.deterministic = True
        class ReproducibleTrainer:
            pass

# =============================================================================
# 1. IMPORTS AND UTILITIES
# =============================================================================

# Import tokenizer from dedicated module
try:
    from model.tokenizer_utils import DynamoTokenizer
except ImportError:
    from tokenizer_utils import DynamoTokenizer

# Import model from dedicated module
try:
    from model.model import DynamoModel
except ImportError:
    from model import DynamoModel

# =============================================================================
# 2. DATA PROCESSING
# =============================================================================

def parse_causal_trace(trace_str: str, node_list: List[str]) -> torch.Tensor:
    """Parse causal trace string into edge index tensor"""
    pattern = r'\[([^\]]+)\]'
    node_sequence = re.findall(pattern, trace_str)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    
    edges = []
    for i in range(len(node_sequence) - 1):
        src_node = node_sequence[i].strip()
        dst_node = node_sequence[i + 1].strip()
        src_idx = node_to_index.get(src_node)
        dst_idx = node_to_index.get(dst_node)
        if src_idx is not None and dst_idx is not None:
            edges.append([src_idx, dst_idx])
    
    if not edges:
        return torch.tensor([[], []], dtype=torch.long)
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def extract_all_nodes(raw_data: List[Dict]) -> List[str]:
    """Extract all unique nodes from causal traces"""
    all_nodes = set()
    pattern = r'\[([^\]]+)\]'
    for item in raw_data:
        nodes = re.findall(pattern, item["causal_trace"])
        for node in nodes:
            all_nodes.add(node.strip())
    return sorted(list(all_nodes))

def parse_date_to_time(date_str: str) -> float:
    """Convert date to normalized time"""
    try:
        year, month, day = map(int, date_str.split('-'))
        return year + (month - 1) / 12 + (day - 1) / 365
    except:
        return 2025.0  # Default fallback

class DynamoDataset(Dataset):
    def __init__(self, processed_data: List[Dict]):
        self.data = processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    time = torch.stack([item['time'] for item in batch])
    edge_indices = [item['edge_index'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'time': time,
        'edge_indices': edge_indices
    }

def process_data(data_file: str, model_name: str, use_temporal_split: bool = True, test_size: float = 0.2):
    """Complete data processing pipeline with proper temporal splitting"""
    print("📊 Processing data...")
    
    # Load raw data
    with open(data_file, 'r') as f:
        raw_data = json.load(f)
    print(f"   Loaded {len(raw_data)} samples")
    
    # Initialize tokenizer
    tokenizer = DynamoTokenizer(model_name)
    
    # Extract nodes
    node_list = extract_all_nodes(raw_data)
    print(f"   Found {len(node_list)} unique nodes")
    
    if use_temporal_split:
        # Create temporal splits to prevent data leakage
        try:
            from model.temporal_data_splitter import create_batch_based_splits, validate_temporal_split
        except ImportError:
            from temporal_data_splitter import create_batch_based_splits, validate_temporal_split
        
        print("   Using temporal splitting to prevent data leakage...")
        raw_train, raw_val, raw_test = create_batch_based_splits(raw_data)
        
        # Validate temporal consistency
        if not validate_temporal_split(raw_train, raw_val, raw_test):
            raise ValueError("Temporal split validation failed!")
        
        # Process each split
        train_data = []
        val_data = []
        test_data = []
        
        for split_data, processed_split, split_name in [
            (raw_train, train_data, "training"),
            (raw_val, val_data, "validation"), 
            (raw_test, test_data, "test")
        ]:
            for item in tqdm(split_data, desc=f"   Processing {split_name} samples"):
                try:
                    tokenized = tokenizer.tokenize_qa_pair(item["question"], item["answer"])
                    time_val = parse_date_to_time(item["date"])
                    edge_index = parse_causal_trace(item["causal_trace"], node_list)
                    
                    data_point = {
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                        "labels": tokenized["labels"],
                        "time": torch.tensor([time_val], dtype=torch.float32),
                        "edge_index": edge_index,
                        "question": item["question"],
                        "answer": item["answer"],
                        "date": item["date"],  # Keep original date for validation
                        "timestamp": time_val  # Add timestamp for evaluation compatibility
                    }
                    processed_split.append(data_point)
                except Exception as e:
                    print(f"   Failed to process {split_name} item: {e}")
                    continue
        
        print(f"   Temporal split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data, node_list, tokenizer
        
    else:
        # Fallback to random split (not recommended for temporal data)
        print("   ⚠️  Using random split - may cause data leakage for temporal data!")
        
        processed_data = []
        for item in tqdm(raw_data, desc="   Processing samples"):
            try:
                tokenized = tokenizer.tokenize_qa_pair(item["question"], item["answer"])
                time_val = parse_date_to_time(item["date"])
                edge_index = parse_causal_trace(item["causal_trace"], node_list)
                
                data_point = {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["labels"],
                    "time": torch.tensor([time_val], dtype=torch.float32),
                    "edge_index": edge_index,
                    "question": item["question"],
                    "answer": item["answer"],
                    "date": item["date"],
                    "timestamp": time_val
                }
                processed_data.append(data_point)
            except Exception as e:
                print(f"   Failed to process item: {e}")
                continue
        
        # Random train-test split
        train_data, test_data = train_test_split(
            processed_data, test_size=test_size, random_state=42
        )
        
        # Create empty validation set for consistency
        val_data = []
        
        print(f"   Random split: {len(train_data)} train, {len(test_data)} test")
        return train_data, val_data, test_data, node_list, tokenizer

# =============================================================================
# 3. MODEL UTILITIES (using imported model)
# =============================================================================

# =============================================================================
# 4. TRAINING LOGIC (using imported trainer)
# =============================================================================

# Import training utilities
try:
    from model.train import DynamoTrainer
except ImportError:
    from train import DynamoTrainer

# =============================================================================
# 5. INFERENCE
# =============================================================================

def test_inference(model, tokenizer, node_list, sample_data, device):
    """Test inference with a sample"""
    print("🧠 Testing inference...")
    
    if not sample_data:
        print("   No sample data available")
        return
    
    sample = sample_data[0]
    question = sample['question']
    true_answer = sample['answer']
    
    print(f"   Question: {question[:60]}...")
    print(f"   True Answer: {true_answer}")
    
    try:
        # Prepare inputs
        inputs = tokenizer.tokenize_qa_pair(question, None)
        input_ids = inputs['input_ids'].unsqueeze(0).to(device)
        attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
        time_tensor = sample['time'].unsqueeze(0).to(device)
        edge_index = sample['edge_index'].to(device)
        
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                time=time_tensor,
                edge_indices=[edge_index],
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.tokenizer.eos_token_id,
                eos_token_id=tokenizer.tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated_ids[0])
        predicted_answer = tokenizer.extract_answer(generated_text)
        
        print(f"   Predicted: {predicted_answer}")
        print("   ✅ Inference test successful!")
        
    except Exception as e:
        print(f"   ⚠️  Inference test failed: {e}")

# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Complete DYNAMO Pipeline")
    parser.add_argument('--data', type=str, default='data/dynamodata.json',
                        help="Input data file")
    parser.add_argument('--transformer', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Transformer model")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Training epochs")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device (auto/cuda/cpu)")
    parser.add_argument('--quick_test', action='store_true',
                        help="Quick test mode")
    
    args = parser.parse_args()
    
    # Setup reproducibility FIRST
    set_all_seeds(42)
    setup_deterministic_training()
    
    # Quick test adjustments
    if args.quick_test:
        args.epochs = 1
        args.batch_size = 1
        args.learning_rate = 1e-4
        print("🏃 Quick test mode enabled")
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 DYNAMO Pipeline Starting")
    print(f"   Device: {device}")
    print(f"   Data: {args.data}")
    print(f"   Transformer: {args.transformer}")
    print(f"   Reproducibility: Enabled (seed=42)")
    
    # Check data file
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    start_time = time.time()
    
    try:
        # Step 1: Process data with temporal splitting
        train_data, val_data, test_data, node_list, tokenizer = process_data(
            args.data, args.transformer, use_temporal_split=True
        )
        
        # Step 2: Create datasets and loaders
        train_dataset = DynamoDataset(train_data)
        val_dataset = DynamoDataset(val_data) if val_data else None
        test_dataset = DynamoDataset(test_data)
        
        # Create reproducible data loaders
        def worker_init_fn(worker_id):
            np.random.seed(42 + worker_id)
            random.seed(42 + worker_id)
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=True, collate_fn=collate_fn,
            num_workers=0,  # Disable multiprocessing for reproducibility
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Use validation set if available, otherwise create a small validation split from training
        if val_dataset:
            eval_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, 
                shuffle=False, collate_fn=collate_fn,
                num_workers=0, worker_init_fn=worker_init_fn
            )
            print("   Using validation set for training evaluation")
        else:
            # Create a small validation split from training data to avoid test set leakage
            val_size = max(1, len(train_data) // 10)  # 10% of training data
            train_subset = train_data[:-val_size]
            val_subset = train_data[-val_size:]
            
            train_dataset = DynamoDataset(train_subset)
            val_dataset = DynamoDataset(val_subset)
            
            # Update train_loader with reduced training data
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, 
                shuffle=True, collate_fn=collate_fn,
                num_workers=0, worker_init_fn=worker_init_fn,
                generator=torch.Generator().manual_seed(42)
            )
            
            eval_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, 
                shuffle=False, collate_fn=collate_fn,
                num_workers=0, worker_init_fn=worker_init_fn
            )
            print(f"   Created validation split from training data: {len(train_subset)} train, {len(val_subset)} val")
            print("   ✅ Test set protected from training leakage")
        
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, 
            shuffle=False, collate_fn=collate_fn,
            num_workers=0, worker_init_fn=worker_init_fn
        )
        
        # Step 3: Create model
        config = {
            'transformer': args.transformer,  # Add missing transformer config
            'use_time2vec': True,
            'use_gnn': True,
            'time2vec_dim': 64,
            'node_dim': 128,
            'gnn_output_dim': 64,
            'device': device,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size
        }
        
        print("🏗️  Creating model...")
        model = DynamoModel(config)
        model._node_list = node_list  # Store for saving
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Step 4: Train model using proper trainer
        print("🚂 Initializing trainer...")
        trainer = DynamoTrainer(model, train_loader, eval_loader, node_list, config)
        
        print("🚂 Starting training...")
        training_stats = trainer.train()
        
        # Extract losses for compatibility
        train_losses = training_stats['train_losses']
        eval_losses = training_stats['eval_losses']
        
        # Step 5: Test inference
        test_inference(model, tokenizer, node_list, test_data, device)
        
        # Save final results
        final_stats = {
            'train_losses': train_losses,
            'eval_losses': eval_losses,
            'config': config,
            'total_time': time.time() - start_time
        }
        
        with open('training_stats.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"   Training time: {time.time() - start_time:.1f} seconds")
        print(f"   Best model saved: best_dynamo_model.pt")
        print(f"   Training stats: training_stats.json")
        print(f"\n🧠 To use your trained model:")
        print(f"   python -c \"")
        print(f"import torch")
        print(f"checkpoint = torch.load('best_dynamo_model.pt')")
        print(f"print('Model ready for inference!')\"")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
