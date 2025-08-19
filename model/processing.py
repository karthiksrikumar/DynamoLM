import json
import torch
from transformers import AutoTokenizer
import re
from typing import List

def parse_causal_trace(trace_str: str, node_list: List[str]) -> torch.Tensor:
    pattern = r'\[([^\]]+)\]'
    node_sequence = re.findall(pattern, trace_str)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    edges = []
    
    for i in range(len(node_sequence) - 1):
        src = node_to_index.get(node_sequence[i])
        dst = node_to_index.get(node_sequence[i+1])
        if src is not None and dst is not None:
            edges.append([src, dst])
    
    if not edges:
        return torch.tensor([[], []], dtype=torch.long)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def preprocess_qa_data(input_file: str, output_file: str, model_name: str = "llama-2-7b"):
    with open(input_file, 'r') as f:
        raw_data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_nodes = set()
    pattern = r'\[([^\]]+)\]'
    for item in raw_data:
        trace_str = item["causal_trace"]
        nodes_in_trace = re.findall(pattern, trace_str)
        all_nodes.update(nodes_in_trace)
    
    global_node_list = list(all_nodes)
    print(f"Found {len(global_node_list)} unique nodes.")
    
    processed_data_list = []
    
    for item in raw_data:
        question = item["question"]
        answer = item["answer"]
        
        encoded_qa = tokenizer(
            question, 
            answer, 
            padding="max_length", 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )
        
        input_ids = encoded_qa["input_ids"][0]
        
        sep_token_id = tokenizer.sep_token_id
        sep_positions = torch.where(input_ids == sep_token_id)[0]
        
        if len(sep_positions) >= 2:
            answer_start_idx = sep_positions[0] + 1
            answer_end_idx = sep_positions[1]
            
            labels = input_ids.clone()
            labels[:answer_start_idx] = -100
        else:
            labels = input_ids.clone()
        
        date_str = item["date"]
        year, month, day = map(int, date_str.split('-'))
        time = year + (month - 1) / 12 + (day - 1) / 365
        
        edge_index = parse_causal_trace(item["causal_trace"], global_node_list)
        
        data_point = {
            "input_ids": encoded_qa["input_ids"].flatten(),
            "attention_mask": encoded_qa["attention_mask"].flatten(),
            "labels": labels,
            "time": torch.tensor([time], dtype=torch.float),
            "edge_index": edge_index,
            "original_question": question,
            "original_answer": answer
        }
        processed_data_list.append(data_point)
    
    torch.save({
        "processed_data": processed_data_list,
        "node_list": global_node_list
    }, output_file)
    print(f"Processed QA data saved to {output_file}")

if __name__ == "__main__":
    preprocess_qa_data("dynamodata.json", "processed_dynamodata_qa.pt")
