# inference.py - Fixed inference script
import torch
from model import create_dynamo_model
from tokenizer_utils import DynamoTokenizer
from processing import parse_causal_trace
import argparse
import json
import os

def load_model_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config and node list
    config = checkpoint['config']
    node_list = checkpoint['node_list']
    
    # Create model
    model = create_dynamo_model(config['transformer'], **config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Global step: {checkpoint.get('global_step', 'Unknown')}")
    print(f"Best loss: {checkpoint.get('best_loss', 'Unknown')}")
    
    return model, node_list, config

def generate_answer(model, tokenizer, question: str, date: str, causal_trace: str, 
                   node_list: list, device: str = 'cuda', max_length: int = 512, 
                   temperature: float = 0.7) -> str:
    """Generate answer for a given question"""
    
    # Parse inputs
    edge_index = parse_causal_trace(causal_trace, node_list)
    # Use the same time parsing function as training
    def parse_date_to_time(date_str: str) -> float:
        """Convert date to normalized time (same as fullpipeline.py)"""
        try:
            year, month, day = map(int, date_str.split('-'))
            return year + (month - 1) / 12 + (day - 1) / 365
        except:
            return 2025.0  # Default fallback
    
    time_value = parse_date_to_time(date)
    
    # Tokenize question (without answer for inference)
    inputs = tokenizer.tokenize_qa_pair(question, None)
    
    # Move to device
    input_ids = inputs['input_ids'].unsqueeze(0).to(device)
    attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
    time_tensor = torch.tensor([time_value], dtype=torch.float).to(device)
    edge_index = edge_index.to(device)
    
    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  time: {time_tensor.shape}")
    print(f"  edge_index: {edge_index.shape}")
    
    # Generate answer
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                time=time_tensor,
                edge_indices=edge_index,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.tokenizer.eos_token_id,
                eos_token_id=tokenizer.tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0])
            
            # Extract answer part
            answer = tokenizer.extract_answer(generated_text)
            
            return answer.strip()
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error: {str(e)}"

def interactive_inference(model, tokenizer, node_list, device):
    """Interactive inference mode"""
    print("\n=== Interactive Inference Mode ===")
    print("Enter 'quit' to exit")
    
    while True:
        try:
            # Get inputs
            question = input("\nEnter question: ").strip()
            if question.lower() == 'quit':
                break
                
            date = input("Enter date (YYYY-MM-DD): ").strip()
            if date.lower() == 'quit':
                break
                
            print("Enter causal trace (format: [Node1] → [Node2] → [Node3]):")
            causal_trace = input().strip()
            if causal_trace.lower() == 'quit':
                break
            
            print(f"\nProcessing...")
            print(f"Question: {question}")
            print(f"Date: {date}")
            print(f"Causal trace: {causal_trace}")
            
            # Generate answer
            answer = generate_answer(
                model, tokenizer, question, date, causal_trace, 
                node_list, device
            )
            
            print(f"\nGenerated Answer: {answer}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def batch_inference(model, tokenizer, node_list, input_file, output_file, device):
    """Batch inference from JSON file"""
    print(f"\n=== Batch Inference Mode ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    for i, item in enumerate(data):
        print(f"\nProcessing item {i+1}/{len(data)}")
        
        question = item['question']
        date = item['date']
        causal_trace = item['causal_trace']
        true_answer = item.get('answer', 'Unknown')
        
        print(f"Question: {question[:50]}...")
        
        # Generate answer
        predicted_answer = generate_answer(
            model, tokenizer, question, date, causal_trace, 
            node_list, device
        )
        
        result = {
            'question': question,
            'date': date,
            'causal_trace': causal_trace,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer
        }
        
        results.append(result)
        
        print(f"True: {true_answer}")
        print(f"Predicted: {predicted_answer}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with DYNAMO QA model")
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use (auto, cuda, cpu)")
    
    # Inference mode
    parser.add_argument('--mode', type=str, choices=['single', 'interactive', 'batch'],
                        default='single', help="Inference mode")
    
    # Single inference arguments
    parser.add_argument('--question', type=str,
                        help="Question to answer (for single mode)")
    parser.add_argument('--date', type=str,
                        help="Date for the question (YYYY-MM-DD)")
    parser.add_argument('--causal_trace', type=str,
                        help="Causal trace for the question")
    
    # Batch inference arguments
    parser.add_argument('--input_file', type=str,
                        help="Input JSON file for batch inference")
    parser.add_argument('--output_file', type=str, default='inference_results.json',
                        help="Output file for batch inference results")
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum generation length")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Generation temperature")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Load model
    model, node_list, config = load_model_checkpoint(args.checkpoint, device)
    
    # Initialize tokenizer
    tokenizer = DynamoTokenizer(config['transformer'])
    
    print(f"\nModel ready for inference")
    print(f"Number of nodes: {len(node_list)}")
    
    # Run inference based on mode
    if args.mode == 'single':
        if not all([args.question, args.date, args.causal_trace]):
            raise ValueError("Single mode requires --question, --date, and --causal_trace")
        
        print(f"\n=== Single Inference ===")
        print(f"Question: {args.question}")
        print(f"Date: {args.date}")
        print(f"Causal trace: {args.causal_trace}")
        
        answer = generate_answer(
            model, tokenizer, args.question, args.date, args.causal_trace,
            node_list, device, args.max_length, args.temperature
        )
        
        print(f"\nGenerated Answer: {answer}")
        
    elif args.mode == 'interactive':
        interactive_inference(model, tokenizer, node_list, device)
        
    elif args.mode == 'batch':
        if not args.input_file:
            raise ValueError("Batch mode requires --input_file")
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        batch_inference(model, tokenizer, node_list, args.input_file, 
                       args.output_file, device)

if __name__ == "__main__":
    main()
