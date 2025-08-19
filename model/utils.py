# utils.py
import torch

def get_linear_scheduler(optimizer, num_training_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
