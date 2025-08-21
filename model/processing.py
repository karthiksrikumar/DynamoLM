# processing.py
import json
import torch
from tokenizer_utils import DynamoTokenizer
import re
from typing import List, Dict, Any
import os

def parse_causal_trace(trace_str: str, node_list: List[str]) -> torch.Tensor:
    """Parse causal trace string into edge index tensor"""
    pattern = r'\[([^\]]+)\]'
    node_sequence = re.findall(pattern, trace_str)
    
    # Create node to index mapping
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    
    edges = []
    
    # Create edges between consecutive nodes in the trace
    for i in range(len(node_sequence) - 1):
        src_node = node_sequence[i].strip()
        dst_node = node_sequence[i + 1].strip()
        
        src_idx = node_to_index.get(src_node)
        dst_idx = node_to_index.get(dst_node)
        
        if src_idx is not None and dst_idx is not None:
            edges.append([src_idx, dst_idx])
    
    if not edges:
        # Return empty edge index with correct shape [2, 0]
        return torch.tensor([[], []], dtype=torch.long)
    
    # Convert to PyTorch Geometric format: [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def extract_all_nodes(raw_data: List[Dict]) -> List[str]:
    """Extract all unique nodes from the dataset"""
    all_nodes = set()
    pattern = r'\[([^\]]+)\]'
    
    for item in raw_data:
        trace_str = item["causal_trace"]
        nodes_in_trace = re.findall(pattern, trace_str)
        # Clean and add nodes
        for node in nodes_in_trace:
            all_nodes.add(node.strip())
    
    return sorted(list(all_nodes))  # Sort for consistency

def parse_date_to_normalized_time(date_str: str) -> float:
    """Convert date string to normalized time value"""
    try:
        year, month, day = map(int, date_str.split('-'))
        # Normalize time as year + fractional year
        time_value = year + (month - 1) / 12 + (day - 1) / 365
        return time_value
    except Exception as e:
        raise ValueError(f"Invalid date format '{date_str}': {e}")

def preprocess_qa_data(input_file: str, output_file: str, model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Preprocess QA data for DYNAMO model training.
    
    Args:
        input_file: Path to JSON file with raw QA data
        output_file: Path to save processed .pt file
        model_name: HuggingFace model name for tokenizer
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    # Load raw data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} QA pairs")
    
    # Initialize tokenizer
    print(f"Initializing tokenizer for {model_name}...")
    try:
        tokenizer = DynamoTokenizer(model_name)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("Falling back to a simpler model...")
        tokenizer = DynamoTokenizer("gpt2")  # Fallback option
    
    # Extract all unique nodes
    print("Extracting unique nodes from causal traces...")
    global_node_list = extract_all_nodes(raw_data)
    print(f"Found {len(global_node_list)} unique nodes")
    
    # Process each QA pair
    processed_data_list = []
    failed_items = 0
    
    print("Processing QA pairs...")
    for i, item in enumerate(raw_data):
        try:
            question = item["question"]
            answer = item["answer"]
            date_str = item["date"]
            causal_trace = item["causal_trace"]
            
            # Tokenize QA pair
            tokenized = tokenizer.tokenize_qa_pair(question, answer)
            
            # Parse date to normalized time
            time_value = parse_date_to_normalized_time(date_str)
            
            # Parse causal trace to edge index
            edge_index = parse_causal_trace(causal_trace, global_node_list)
            
            # Create data point
            data_point = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["labels"],
                "time": torch.tensor([time_value], dtype=torch.float32),
                "edge_index": edge_index,
                "original_question": question,
                "original_answer": answer,
                "original_date": date_str,
                "original_trace": causal_trace
            }
            
            processed_data_list.append(data_point)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(raw_data)} items")
                
        except Exception as e:
            print(f"Failed to process item {i}: {e}")
            failed_items += 1
            continue
    
    print(f"Successfully processed {len(processed_data_list)} items")
    if failed_items > 0:
        print(f"Failed to process {failed_items} items")
    
    # Create final data structure
    processed_data = {
        "processed_data": processed_data_list,
        "node_list": global_node_list,
        "num_samples": len(processed_data_list),
        "num_nodes": len(global_node_list),
        "model_name": model_name
    }
    
    # Save processed data
    print(f"Saving processed data to {output_file}...")
    torch.save(processed_data, output_file)
    
    # Print summary
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total samples: {len(processed_data_list)}")
    print(f"Unique nodes: {len(global_node_list)}")
    print(f"Failed items: {failed_items}")
    print(f"Output file: {output_file}")
    
    # Print sample node list (first 10)
    print(f"\nSample nodes: {global_node_list[:10]}")
    
    # Print sample data point info
    if processed_data_list:
        sample = processed_data_list[0]
        print(f"\nSample data point:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Edge index shape: {sample['edge_index'].shape}")
        print(f"  Time value: {sample['time'].item():.4f}")
        print(f"  Question: {sample['original_question'][:50]}...")
    
    print(f"\nProcessed data saved to {output_file}")
    return processed_data

def validate_processed_data(data_path: str) -> bool:
    """Validate that processed data is compatible with the model"""
    print(f"Validating processed data from {data_path}...")
    
    try:
        data_dict = torch.load(data_path)
        required_keys = ["processed_data", "node_list"]
        
        for key in required_keys:
            if key not in data_dict:
                print(f"Missing required key: {key}")
                return False
        
        processed_data = data_dict["processed_data"]
        node_list = data_dict["node_list"]
        
        print(f"Found {len(processed_data)} samples and {len(node_list)} nodes")
        
        # Validate sample structure
        if processed_data:
            sample = processed_data[0]
            required_sample_keys = ["input_ids", "attention_mask", "labels", "time", "edge_index"]
            
            for key in required_sample_keys:
                if key not in sample:
                    print(f"Missing required sample key: {key}")
                    return False
            
            # Check tensor shapes
            print(f"Sample tensor shapes:")
            print(f"  input_ids: {sample['input_ids'].shape}")
            print(f"  attention_mask: {sample['attention_mask'].shape}")
            print(f"  labels: {sample['labels'].shape}")
            print(f"  time: {sample['time'].shape}")
            print(f"  edge_index: {sample['edge_index'].shape}")
            
            # Validate edge index format
            edge_index = sample['edge_index']
            if edge_index.numel() > 0:
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    print(f"Invalid edge index shape: {edge_index.shape}, expected [2, num_edges]")
                    return False
                
                max_node_idx = edge_index.max().item()
                if max_node_idx >= len(node_list):
                    print(f"Edge index references node {max_node_idx} but only {len(node_list)} nodes exist")
                    return False
        
        print("Data validation passed!")
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess DYNAMO QA data")
    parser.add_argument('--input', type=str, default='data/dynamodata.json',
                        help="Input JSON file path")
    parser.add_argument('--output', type=str, default='processed_dynamodata_qa.pt',
                        help="Output processed data file path")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Model name for tokenizer")
    parser.add_argument('--validate', action='store_true',
                        help="Only validate existing processed data")
    
    args = parser.parse_args()
    
    if args.validate:
        if os.path.exists(args.output):
            validate_processed_data(args.output)
        else:
            print(f"File {args.output} not found")
    else:
        # Process the data
        processed_data = preprocess_qa_data(args.input, args.output, args.model)
        
        # Validate the processed data
        validate_processed_data(args.output)
