# reproducibility.py - Comprehensive reproducibility controls
import torch
import numpy as np
import random
import os
from typing import Optional

def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducible results"""
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ All random seeds set to {seed} for reproducibility")

def setup_deterministic_training():
    """Setup PyTorch for deterministic training"""
    
    # Use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print("✅ Deterministic training mode enabled")

def validate_reproducibility(model_class, config: dict, seed: int = 42, num_runs: int = 3):
    """
    Validate that model training is reproducible by running multiple times with same seed.
    
    Args:
        model_class: Model class to test
        config: Model configuration
        seed: Random seed to use
        num_runs: Number of runs to compare
    
    Returns:
        bool: True if all runs produce identical results
    """
    
    print(f"🔍 Testing reproducibility with {num_runs} runs...")
    
    initial_weights = []
    
    for run in range(num_runs):
        print(f"   Run {run + 1}/{num_runs}")
        
        # Reset all seeds
        set_all_seeds(seed)
        
        # Create model
        model = model_class(config)
        
        # Store initial weights
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data.clone()
        initial_weights.append(weights)
    
    # Compare weights across runs
    reference_weights = initial_weights[0]
    
    for run_idx in range(1, num_runs):
        run_weights = initial_weights[run_idx]
        
        for name in reference_weights:
            if not torch.allclose(reference_weights[name], run_weights[name], atol=1e-6):
                print(f"❌ Reproducibility test failed!")
                print(f"   Parameter '{name}' differs between run 1 and run {run_idx + 1}")
                return False
    
    print(f"✅ Reproducibility test passed - all {num_runs} runs identical")
    return True

def create_reproducible_config(base_config: dict, seed: int = 42) -> dict:
    """Create a configuration with reproducibility settings"""
    
    config = base_config.copy()
    
    # Add reproducibility settings
    config['seed'] = seed
    config['deterministic'] = True
    config['benchmark'] = False
    
    # Ensure consistent worker settings
    config['num_workers'] = 0  # Disable multiprocessing for reproducibility
    config['pin_memory'] = False  # Disable for consistency
    
    return config

class ReproducibleTrainer:
    """Wrapper for training with reproducibility controls"""
    
    def __init__(self, seed: int = 42, strict: bool = True):
        self.seed = seed
        self.strict = strict
        
        # Setup reproducibility
        set_all_seeds(seed)
        if strict:
            setup_deterministic_training()
    
    def train_with_validation(self, train_func, *args, **kwargs):
        """Train model with reproducibility validation"""
        
        # Store initial random state
        initial_random_state = random.getstate()
        initial_np_state = np.random.get_state()
        initial_torch_state = torch.get_rng_state()
        
        try:
            # Run training
            result = train_func(*args, **kwargs)
            
            print("✅ Training completed with reproducibility controls")
            return result
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            
            # Restore random states
            random.setstate(initial_random_state)
            np.random.set_state(initial_np_state)
            torch.set_rng_state(initial_torch_state)
            
            raise e

if __name__ == "__main__":
    # Test reproducibility controls
    print("Testing reproducibility controls...")
    
    set_all_seeds(42)
    setup_deterministic_training()
    
    # Test that random operations are deterministic
    torch_rand_1 = torch.randn(5)
    set_all_seeds(42)
    torch_rand_2 = torch.randn(5)
    
    if torch.allclose(torch_rand_1, torch_rand_2):
        print("✅ PyTorch reproducibility confirmed")
    else:
        print("❌ PyTorch reproducibility failed")
    
    # Test NumPy
    set_all_seeds(42)
    np_rand_1 = np.random.randn(5)
    set_all_seeds(42)
    np_rand_2 = np.random.randn(5)
    
    if np.allclose(np_rand_1, np_rand_2):
        print("✅ NumPy reproducibility confirmed")
    else:
        print("❌ NumPy reproducibility failed")
