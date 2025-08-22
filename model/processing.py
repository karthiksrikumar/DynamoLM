# processing.py - Fixed version for dynamodata.json
import json
import re
from typing import List, Dict

def load_dynamodata_json(input_file: str = "data/dynamodata.json") -> List[Dict]:
    print(f"Loading dynamodata from {input_file}...")
    
    with open(input_file, 'r') as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print(f"Loaded single array with {len(data)} items")
            return data
        else:
            print(f"Warning: Expected list, got {type(data)}")
            return []
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return []

def validate_data_sample(sample: Dict, index: int = 0) -> bool:
    required_keys = ["question", "date", "answer", "causal_trace"]
    for key in required_keys:
        if key not in sample:
            print(f"Sample {index}: Missing key '{key}'")
            return False
        if not isinstance(sample[key], str) or not sample[key].strip():
            print(f"Sample {index}: Key '{key}' is empty or invalid")
            return False

    # Date validation
    try:
        year, month, day = map(int, sample["date"].split("-"))
        if not (1900 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31):
            print(f"Sample {index}: Invalid date {sample['date']}")
            return False
    except:
        print(f"Sample {index}: Could not parse date '{sample['date']}'")
        return False

    # causal_trace format
    if "[" not in sample["causal_trace"] or "]" not in sample["causal_trace"]:
        print(f"Sample {index}: Invalid causal trace '{sample['causal_trace']}'")
        return False
    
    return True

def parse_causal_trace(trace_str: str):
    """Convert causal trace string into edges for graph construction"""
    pattern = r'\[([^\]]+)\]'
    node_sequence = re.findall(pattern, trace_str)
    node_to_index = {node: idx for idx, node in enumerate(node_sequence)}
    
    edges = []
    for i in range(len(node_sequence) - 1):
        src, dst = node_sequence[i].strip(), node_sequence[i+1].strip()
        if src in node_to_index and dst in node_to_index:
            edges.append([node_to_index[src], node_to_index[dst]])
    
    return edges
