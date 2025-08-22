#!/usr/bin/env python3
# run_pipeline.py - Complete pipeline runner for DYNAMO
"""
Complete pipeline runner for the DYNAMO QA model.
This script runs the entire pipeline from data preprocessing to training and inference.
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS: {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED: {description} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False

def check_requirements():
    """Check if required files and dependencies exist"""
    print("🔍 Checking requirements...")
    
    required_files = [
        'data/dynamodata.json',
        'model/processing.py',
        'model/model.py',
        'model/train.py',
        'model/inference.py',
        'model/tokenizer_utils.py',
        'model/utils.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Check if data file has content
    try:
        with open('data/dynamodata.json', 'r') as f:
            data = json.load(f)
        if len(data) == 0:
            print("❌ Data file is empty")
            return False
        print(f"✅ Found {len(data)} data samples")
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return False
    
    print("✅ All requirements check passed")
    return True

def run_preprocessing(args):
    """Run data preprocessing"""
    print(f"\n📊 Starting data preprocessing...")
    
    cmd = f"""python model/processing.py \
        --input {args.input_data} \
        --output_dir {args.processed_data_dir} \
        --model {args.transformer} \
        --test_size {args.test_size}"""
    
    return run_command(cmd, "Data Preprocessing")

def run_training(args):
    """Run model training"""
    print(f"\n🚂 Starting model training...")
    
    train_data = os.path.join(args.processed_data_dir, "train_data.pt")
    test_data = os.path.join(args.processed_data_dir, "test_data.pt")
    
    cmd = f"""python model/train.py \
        --train_data {train_data} \
        --test_data {test_data} \
        --transformer {args.transformer} \
        --batch_size {args.batch_size} \
        --learning_rate {args.learning_rate} \
        --epochs {args.epochs} \
        --device {args.device} \
        --output_dir {args.checkpoint_dir} \
        --warmup_steps {args.warmup_steps} \
        --save_steps {args.save_steps} \
        --eval_steps {args.eval_steps}"""
    
    return run_command(cmd, "Model Training")

def run_inference_test(args):
    """Run a quick inference test"""
    print(f"\n🧠 Testing inference...")
    
    # Use the first sample from data for testing
    try:
        with open(args.input_data, 'r') as f:
            data = json.load(f)
        
        sample = data[0]
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pt")
        
        if not os.path.exists(checkpoint_path):
            print("❌ No trained model found for inference test")
            return False
        
        cmd = f"""python model/inference.py \
            --checkpoint {checkpoint_path} \
            --mode single \
            --question "{sample['question']}" \
            --date "{sample['date']}" \
            --causal_trace "{sample['causal_trace']}" \
            --device {args.device}"""
        
        return run_command(cmd, "Inference Test")
        
    except Exception as e:
        print(f"❌ Error setting up inference test: {e}")
        return False

def create_config_file(args):
    """Create a configuration file for the run"""
    config = {
        "experiment_name": f"dynamo_run_{int(time.time())}",
        "transformer": args.transformer,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "test_size": args.test_size,
        "warmup_steps": args.warmup_steps,
        "device": args.device,
        "input_data": args.input_data,
        "processed_data_dir": args.processed_data_dir,
        "checkpoint_dir": args.checkpoint_dir
    }
    
    config_path = os.path.join(args.checkpoint_dir, "run_config.json")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Configuration saved to {config_path}")
    return config_path

def print_summary(args, success_steps):
    """Print pipeline execution summary"""
    print(f"\n{'='*60}")
    print("📋 PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    steps = [
        ("Requirements Check", "requirements" in success_steps),
        ("Data Preprocessing", "preprocessing" in success_steps),
        ("Model Training", "training" in success_steps),
        ("Inference Test", "inference" in success_steps)
    ]
    
    for step_name, success in steps:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{step_name:<20}: {status}")
    
    print(f"\nConfiguration:")
    print(f"  Transformer: {args.transformer}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Device: {args.device}")
    
    if "training" in success_steps:
        checkpoint_dir = Path(args.checkpoint_dir)
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            print(f"\nSaved Models:")
            for checkpoint in checkpoints:
                size_mb = checkpoint.stat().st_size / (1024*1024)
                print(f"  {checkpoint.name}: {size_mb:.1f} MB")
    
    print(f"\n📁 Output Directory: {args.checkpoint_dir}")
    
    if all(step[1] for step in steps):
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"You can now run inference with:")
        print(f"python model/inference.py --checkpoint {args.checkpoint_dir}/best_model.pt --mode interactive")
    else:
        print(f"\n⚠️  PIPELINE COMPLETED WITH ERRORS")
        print(f"Please check the error messages above and fix any issues.")

def main():
    parser = argparse.ArgumentParser(description="Complete DYNAMO Pipeline Runner")
    
    # Data arguments
    parser.add_argument('--input_data', type=str, default='data/dynamodata.json',
                        help="Path to input JSON data file")
    parser.add_argument('--processed_data_dir', type=str, default='processed_data',
                        help="Directory for processed data")
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="Fraction of data to use for testing")
    
    # Model arguments
    parser.add_argument('--transformer', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Transformer model to use")
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help="Number of warmup steps")
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help="Directory for saving checkpoints")
    
    # Pipeline control
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help="Skip data preprocessing step")
    parser.add_argument('--skip_training', action='store_true',
                        help="Skip training step")
    parser.add_argument('--skip_inference', action='store_true',
                        help="Skip inference test step")
    parser.add_argument('--quick_test', action='store_true',
                        help="Run with minimal settings for quick testing")
    
    # Advanced arguments
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument('--eval_steps', type=int, default=250,
                        help="Evaluate every N steps")
    
    args = parser.parse_args()
    
    # Quick test mode adjustments
    if args.quick_test:
        args.epochs = 1
        args.batch_size = 1
        args.learning_rate = 1e-4
        args.warmup_steps = 10
        args.save_steps = 50
        args.eval_steps = 25
        print("🏃 Quick test mode enabled - using minimal settings")
    
    # Set device
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Starting DYNAMO Pipeline")
    print(f"Device: {args.device}")
    print(f"Transformer: {args.transformer}")
    
    # Track successful steps
    success_steps = set()
    
    # Step 1: Check requirements
    if check_requirements():
        success_steps.add("requirements")
    else:
        print("❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Create configuration file
    config_path = create_config_file(args)
    
    # Step 2: Data preprocessing
    if not args.skip_preprocessing:
        # Check if processed data already exists
        train_data_path = os.path.join(args.processed_data_dir, "train_data.pt")
        test_data_path = os.path.join(args.processed_data_dir, "test_data.pt")
        
        if os.path.exists(train_data_path) and os.path.exists(test_data_path):
            print(f"📊 Processed data already exists, skipping preprocessing")
            print(f"  Train data: {train_data_path}")
            print(f"  Test data: {test_data_path}")
            success_steps.add("preprocessing")
        else:
            if run_preprocessing(args):
                success_steps.add("preprocessing")
            else:
                print("❌ Preprocessing failed. Check the error messages above.")
                print_summary(args, success_steps)
                sys.exit(1)
    else:
        print("⏭️  Skipping preprocessing step")
        success_steps.add("preprocessing")
    
    # Step 3: Training
    if not args.skip_training:
        # Check if model already exists
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        final_model_path = os.path.join(args.checkpoint_dir, "final_model.pt")
        
        if os.path.exists(best_model_path) or os.path.exists(final_model_path):
            print(f"🚂 Trained model already exists, skipping training")
            if os.path.exists(best_model_path):
                print(f"  Best model: {best_model_path}")
            if os.path.exists(final_model_path):
                print(f"  Final model: {final_model_path}")
            success_steps.add("training")
        else:
            if run_training(args):
                success_steps.add("training")
            else:
                print("❌ Training failed. Check the error messages above.")
                print_summary(args, success_steps)
                sys.exit(1)
    else:
        print("⏭️  Skipping training step")
        success_steps.add("training")
    
    # Step 4: Inference test
    if not args.skip_inference:
        if run_inference_test(args):
            success_steps.add("inference")
        else:
            print("⚠️  Inference test failed, but continuing...")
    else:
        print("⏭️  Skipping inference test step")
        success_steps.add("inference")
    
    # Print final summary
    print_summary(args, success_steps)

if __name__ == "__main__":
    main()
