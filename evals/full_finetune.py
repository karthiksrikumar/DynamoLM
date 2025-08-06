import torch
import torch.nn as nn
from transformers import LlamaModel

class FullFinetuneModel(nn.Module):
    """LLaMA-7B with full fine-tuning."""
    def __init__(self, transformer_name: str):
        """
        Args:
            transformer_name (str): Hugging Face model name (e.g., meta-llama/Llama-2-7b-hf).
        """
        super().__init__()
        self.transformer = LlamaModel.from_pretrained(transformer_name)
        self.hidden_dim = self.transformer.config.hidden_size
        self.output_head = nn.Linear(self.hidden_dim, 10)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): Tokenized input of shape [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes].
        """
        text_emb = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.output_head(text_emb)
        return logits
