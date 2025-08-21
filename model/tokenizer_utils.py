# tokenizer_utils.py
from transformers import AutoTokenizer
import torch
from typing import Dict

class DynamoTokenizer:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_qa_pair(self, question: str, answer: str = None, max_length: int = 512) -> Dict[str, torch.Tensor]:
        if answer:
            text = f"<s> Question: {question} Answer: {answer} </s>"
        else:
            text = f"<s> Question: {question} Answer:"
        
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        if answer:
            input_ids = encoding["input_ids"][0]
            
            # Find where "Answer:" starts
            answer_token = self.tokenizer.encode("Answer:", add_special_tokens=False)[0]
            answer_positions = torch.where(input_ids == answer_token)[0]
            
            if len(answer_positions) > 0:
                answer_start_idx = answer_positions[0] + 1  # +1 to skip the "Answer:" token
                labels = input_ids.clone()
                labels[:answer_start_idx] = -100  # Ignore loss on question part
            else:
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
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def extract_answer(self, generated_text: str) -> str:
        answer_start = generated_text.find("Answer:") + len("Answer:")
        return generated_text[answer_start:].strip()
