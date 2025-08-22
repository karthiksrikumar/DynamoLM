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
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. TOKENIZER UTILITIES
# =============================================================================

class DynamoTokenizer:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            print(f"⚠️  Failed to load {model_name}, using gpt2 fallback")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_qa_pair(self, question: str, answer: str = None, max_length: int = 512) -> Dict[str, torch.Tensor]:
        if answer:
            text = f"<s> Question: {question} Answer: {answer} </s>"
        else:
            text = f"<s> Question: {question} Answer:"
        
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        
        if answer:
            input_ids = encoding["input_ids"][0]
            # Find where "Answer:" starts
            try:
                answer_token = self.tokenizer.encode("Answer:", add_special_tokens=False)[0]
                answer_positions = torch.where(input_ids == answer_token)[0]
                if len(answer_positions) > 0:
                    answer_start_idx = answer_positions[0] + 1
                    labels = input_ids.clone()
                    labels[:answer_start_idx] = -100
                else:
                    labels = input_ids.clone()
            except:
                labels = input_ids.clone()
            
            return {
                "input_ids": input_ids,
                "attention_mask": encoding["attention_mask"][0],
                "labels": labels
            }
        else:
            return {
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0]
            }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def extract_answer(self, generated_text: str) -> str:
        try:
            answer_start = generated_text.find("Answer:") + len("Answer:")
            return generated_text[answer_start:].strip()
        except:
            return generated_text.strip()

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

def process_data(data_file: str, model_name: str, test_size: float = 0.2):
    """Complete data processing pipeline"""
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
    
    # Process samples
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
                "answer": item["answer"]
            }
            processed_data.append(data_point)
        except Exception as e:
            print(f"   Failed to process item: {e}")
            continue
    
    # Train-test split
    train_data, test_data = train_test_split(
        processed_data, test_size=test_size, random_state=42
    )
    
    print(f"   Split: {len(train_data)} train, {len(test_data)} test")
    
    return train_data, test_data, node_list, tokenizer

# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================

class Time2Vec(nn.Module):
    def __init__(self, dim: int, activation: str = "sin"):
        super().__init__()
        self.dim = dim
        self.activation = activation
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.randn(1))
        
        if dim > 1:
            self.periodic_weights = nn.Parameter(torch.randn(dim - 1))
            self.periodic_biases = nn.Parameter(torch.randn(dim - 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_weight.unsqueeze(0))
        nn.init.zeros_(self.linear_bias)
        if hasattr(self, 'periodic_weights'):
            nn.init.xavier_uniform_(self.periodic_weights.unsqueeze(0))
            nn.init.uniform_(self.periodic_biases, -torch.pi, torch.pi)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.numel() == 0:
            return torch.empty(*t.shape, self.dim, device=t.device, dtype=t.dtype)
            
        original_shape = t.shape
        t_flat = t.flatten().unsqueeze(-1)
        
        linear_component = self.linear_weight * t_flat + self.linear_bias
        
        if self.dim == 1:
            result = linear_component
        else:
            periodic_input = self.periodic_weights * t_flat + self.periodic_biases
            if self.activation == "sin":
                periodic_component = torch.sin(periodic_input)
            else:
                periodic_component = torch.cos(periodic_input)
            result = torch.cat([linear_component, periodic_component], dim=-1)
        
        return result.view(*original_shape, self.dim)

class DynamoModel(nn.Module):
    def __init__(self, transformer_name: str, config: Dict):
        super().__init__()
        self.config = config
        
        # Load transformer
        try:
            self.transformer = LlamaForCausalLM.from_pretrained(
                transformer_name, torch_dtype=torch.float32, device_map=None
            )
        except:
            print(f"⚠️  Failed to load {transformer_name}, trying alternative...")
            try:
                from transformers import GPT2LMHeadModel
                self.transformer = GPT2LMHeadModel.from_pretrained("gpt2")
            except:
                raise RuntimeError("Could not load any transformer model")
        
        self.hidden_dim = self.transformer.config.hidden_size
        
        # Time2Vec
        self.use_time2vec = config.get('use_time2vec', True)
        if self.use_time2vec:
            self.time2vec_dim = config.get('time2vec_dim', 64)
            self.time2vec = Time2Vec(self.time2vec_dim)
        else:
            self.time2vec_dim = 0
        
        # GNN
        self.use_gnn = config.get('use_gnn', True)
        if self.use_gnn:
            self.node_dim = config.get('node_dim', 128)
            self.gnn_output_dim = config.get('gnn_output_dim', 64)
            self.node_embedding = None
            self.gnn_conv1 = GCNConv(self.node_dim, self.gnn_output_dim)
            self.gnn_conv2 = GCNConv(self.gnn_output_dim, self.gnn_output_dim)
        else:
            self.gnn_output_dim = 0
        
        # Fusion layer
        fusion_input_dim = self.hidden_dim + self.time2vec_dim + self.gnn_output_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
    
    def _init_node_embeddings(self, num_nodes: int, device):
        if self.use_gnn and (self.node_embedding is None or self.node_embedding.num_embeddings != num_nodes):
            self.node_embedding = nn.Embedding(num_nodes, self.node_dim).to(device)
            nn.init.xavier_uniform_(self.node_embedding.weight)
    
    def _process_graph(self, edge_indices: List[torch.Tensor], batch_size: int, device: torch.device):
        if not self.use_gnn or not edge_indices:
            return torch.zeros(batch_size, self.gnn_output_dim, device=device)
        
        # Use first edge index for simplicity
        edge_index = edge_indices[0]
        if edge_index.numel() == 0:
            return torch.zeros(batch_size, self.gnn_output_dim, device=device)
        
        num_nodes = edge_index.max().item() + 1
        self._init_node_embeddings(num_nodes, device)
        
        x = self.node_embedding.weight
        x = F.relu(self.gnn_conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.gnn_conv2(x, edge_index)
        
        # Global pooling
        graph_emb = x.mean(dim=0, keepdim=True)
        return graph_emb.expand(batch_size, -1)
    
    def forward(self, input_ids, attention_mask, time=None, edge_indices=None, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Base embeddings
        base_embeddings = self.transformer.get_input_embeddings()(input_ids)
        
        # Additional features
        additional_features = []
        
        # Time features
        if self.use_time2vec and time is not None:
            time_emb = self.time2vec(time)
            time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
            additional_features.append(time_emb)
        
        # Graph features
        if self.use_gnn and edge_indices is not None:
            graph_emb = self._process_graph(edge_indices, batch_size, device)
            graph_emb = graph_emb.unsqueeze(1).expand(-1, seq_len, -1)
            additional_features.append(graph_emb)
        
        # Fuse features
        if additional_features:
            combined_features = torch.cat([base_embeddings] + additional_features, dim=-1)
            fused_embeddings = base_embeddings + self.fusion_layer(combined_features)
        else:
            fused_embeddings = base_embeddings
        
        # Pass through transformer
        outputs = self.transformer(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask, time=None, edge_indices=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Process features same as forward
        base_embeddings = self.transformer.get_input_embeddings()(input_ids)
        additional_features = []
        
        if self.use_time2vec and time is not None:
            time_emb = self.time2vec(time)
            time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
            additional_features.append(time_emb)
        
        if self.use_gnn and edge_indices is not None:
            graph_emb = self._process_graph(edge_indices, batch_size, device)
            graph_emb = graph_emb.unsqueeze(1).expand(-1, seq_len, -1)
            additional_features.append(graph_emb)
        
        if additional_features:
            combined_features = torch.cat([base_embeddings] + additional_features, dim=-1)
            fused_embeddings = base_embeddings + self.fusion_layer(combined_features)
        else:
            fused_embeddings = base_embeddings
        
        return self.transformer.generate(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            **kwargs
        )

# =============================================================================
# 4. TRAINING LOGIC
# =============================================================================

def train_model(model, train_loader, test_loader, config):
    """Complete training function"""
    print("🚂 Training model...")
    
    device = config['device']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )
    
    best_loss = float('inf')
    train_losses = []
    eval_losses = []
    
    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc="   Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            for key in batch:
                if key != 'edge_indices' and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif key == 'edge_indices':
                    batch[key] = [ei.to(device) for ei in batch[key]]
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                time=batch['time'],
                edge_indices=batch['edge_indices'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="   Evaluating"):
                # Move to device
                for key in batch:
                    if key != 'edge_indices' and isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                    elif key == 'edge_indices':
                        batch[key] = [ei.to(device) for ei in batch[key]]
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    time=batch['time'],
                    edge_indices=batch['edge_indices'],
                    labels=batch['labels']
                )
                eval_loss += outputs.loss.item()
        
        avg_eval_loss = eval_loss / len(test_loader)
        eval_losses.append(avg_eval_loss)
        
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Eval Loss:  {avg_eval_loss:.4f}")
        
        # Save best model
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'node_list': getattr(model, '_node_list', []),
                'train_losses': train_losses,
                'eval_losses': eval_losses,
                'best_loss': best_loss
            }, 'best_dynamo_model.pt')
            print("   ✅ Best model saved!")
    
    return model, train_losses, eval_losses

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
    
    # Check data file
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    start_time = time.time()
    
    try:
        # Step 1: Process data
        train_data, test_data, node_list, tokenizer = process_data(
            args.data, args.transformer
        )
        
        # Step 2: Create datasets and loaders
        train_dataset = DynamoDataset(train_data)
        test_dataset = DynamoDataset(test_data)
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=True, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, 
            shuffle=False, collate_fn=collate_fn
        )
        
        # Step 3: Create model
        config = {
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
        model = DynamoModel(args.transformer, config)
        model._node_list = node_list  # Store for saving
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Step 4: Train model
        model, train_losses, eval_losses = train_model(model, train_loader, test_loader, config)
        
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
