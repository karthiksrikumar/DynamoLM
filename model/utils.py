import torch

def get_linear_scheduler(optimizer, num_training_steps: int, warmup_steps: int):
    """
    Create a linear learning rate scheduler with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        num_training_steps (int): Total number of training steps.
        warmup_steps (int): Number of warmup steps.

    Returns:
        lambda: Learning rate multiplier function.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(n_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
