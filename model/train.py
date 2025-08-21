# processing.py
import json
import torch
from tokenizer_utils import DynamoTokenizer
import re
from typing import List, Dict, Any
import os

def parse_causal_trace(trace_str: str, node_list: List[str]) -> torch.Tensor:
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
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def extract_all_nodes(raw_data: List[Dict]) -> List[str]:
    all_nodes = set()
    pattern = r'\[([^\]]+)\]'
    for item in raw_data:
        trace_str = item["causal_trace"]
        nodes_in_trace = re.findall(pattern, trace_str)
        for node in nodes_in_trace:
            all_nodes.add(node.strip())
    return sorted(list(all_nodes))

def parse_date_to_normalized_time(date_str: str) -> float:
    try:
        year, month, day = map(int, date_str.split('-'))
        time_value = year + (month - 1) / 12 + (day - 1) / 365
        return time_value
    except Exception as e:
        raise ValueError(f"Invalid date format '{date_str}': {e}")

def preprocess_qa_data(input_file: str, output_file: str, model_name: str = "meta-llama/Llama-2-7b-hf"):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    with open(input_file, 'r') as f:
        raw_data = json.load(f)
    tokenizer = DynamoTokenizer(model_name)
    global_node_list = extract_all_nodes(raw_data)
    processed_data_list = []
    failed_items = 0
    for i, item in enumerate(raw_data):
        try:
            question = item["question"]
            answer = item["answer"]
            date_str = item["date"]
            causal_trace = item["causal_trace"]
            tokenized = tokenizer.tokenize_qa_pair(question, answer)
            time_value = parse_date_to_normalized_time(date_str)
            edge_index = parse_causal_trace(causal_trace, global_node_list)
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
        except Exception as e:
            failed_items += 1
            continue
    processed_data = {
        "processed_data": processed_data_list,
        "node_list": global_node_list,
        "num_samples": len(processed_data_list),
        "num_nodes": len(global_node_list),
        "model_name": model_name
    }
    torch.save(processed_data, output_file)
    return processed_data

def validate_processed_data(data_path: str) -> bool:
    try:
        data_dict = torch.load(data_path)
        required_keys = ["processed_data", "node_list"]
        for key in required_keys:
            if key not in data_dict:
                return False
        processed_data = data_dict["processed_data"]
        node_list = data_dict["node_list"]
        if processed_data:
            sample = processed_data[0]
            required_sample_keys = ["input_ids", "attention_mask", "labels", "time", "edge_index"]
            for key in required_sample_keys:
                if key not in sample:
                    return False
            edge_index = sample['edge_index']
            if edge_index.numel() > 0:
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    return False
                max_node_idx = edge_index.max().item()
                if max_node_idx >= len(node_list):
                    return False
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess DYNAMO QA data")
    parser.add_argument('--input', type=str, default='data/dynamodata.json', help="Input JSON file path")
    parser.add_argument('--output', type=str, default='processed_dynamodata_qa.pt', help="Output processed data file path")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help="Model name for tokenizer")
    parser.add_argument('--validate', action='store_true', help="Only validate existing processed data")
    args = parser.parse_args()
    if args.validate:
        if os.path.exists(args.output):
            validate_processed_data(args.output)
        else:
            print(f"File {args.output} not found")
    else:
        processed_data = preprocess_qa_data(args.input, args.output, args.model)
        validate_processed_data(args.output)
