import torch
from model import create_dynamo_model
from tokenizer_utils import DynamoTokenizer
from processing import parse_causal_trace
import argparse

def load_model(model_path, config, device='cuda'):
    model = create_dynamo_model(config['transformer_path'], **config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def generate_answer(model, tokenizer, question, time, causal_trace, node_list, device='cuda'):
    edge_index = parse_causal_trace(causal_trace, node_list)
    inputs = tokenizer.tokenize_qa_pair(question, None)
    input_ids = inputs['input_ids'].unsqueeze(0).to(device)
    attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
    time_tensor = torch.tensor([time], dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time=time_tensor,
            edge_indices=edge_index,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    generated_text = tokenizer.decode(generated_ids[0])
    answer = tokenizer.extract_answer(generated_text)
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with DYNAMO QA model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model")
    parser.add_argument('--question', type=str, required=True, help="Question to answer")
    parser.add_argument('--date', type=str, required=True, help="Date for the question (YYYY-MM-DD)")
    parser.add_argument('--causal_trace', type=str, required=True, help="Causal trace for the question")
    parser.add_argument('--node_list', type=list, required=True, help="Global node list from training")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use")
    args = parser.parse_args()
    config = {
        'transformer_path': 'meta-llama/Llama-2-7b-hf',
        'use_time2vec': True,
        'use_gnn': True,
    }
    model = load_model(args.model_path, config, args.device)
    tokenizer = DynamoTokenizer()
    year, month, day = map(int, args.date.split('-'))
    time = year + (month - 1) / 12 + (day - 1) / 365
    answer = generate_answer(model, tokenizer, args.question, time, args.causal_trace, args.node_list, args.device)
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")
