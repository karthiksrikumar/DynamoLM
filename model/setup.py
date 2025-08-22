def check_huggingface_login():
    """Check HuggingFace login status"""
    print("🤗 Checking HuggingFace authentication...")
    
    try:
        result = subprocess.run("huggingface-cli whoami", shell=True, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✅ Logged in as: {username}")
            
            # Test LLaMA-2 access
            print("🔍 Testing LLaMA-2 access...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Ll#!/usr/bin/env python3
# setup.py - Setup script for DYNAMO
"""
Setup script to ensure all dependencies and data are properly configured.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description):
    """Run command and return success status"""
    print(f"🔧 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_gpu():
    """Check GPU availability"""
    print("🔧 Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ Found {gpu_count} GPU(s): {gpu_name} ({memory_gb:.1f}GB)")
            
            if memory_gb < 8:
                print("⚠️  Warning: Less than 8GB GPU memory detected")
                print("   Consider using smaller batch sizes or simpler models")
            return True
        else:
            print("⚠️  No GPU detected - will use CPU (much slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data",
        "model", 
        "processed_data",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}/")
    
    return True

def verify_data_file():
    """Verify data file exists and is valid"""
    print("📊 Checking data file...")
    
    data_file = "data/dynamodata.json"
    
    if not os.path.exists(data_file):
        print(f"⚠️  Data file not found: {data_file}")
        print("Creating sample dynamodata.json with provided data...")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Use the sample data from the documents
        sample_data = [
            {
                "question": "What was the peak sustained wind speed of Hurricane Felix at landfall in Florida?",
                "date": "2025-09-10",
                "answer": "145 mph",
                "causal_trace": "[NHC Landfall Advisory] → [Category 4 Classification] → [Peak Wind Measurement Protocol] → [Advisory #24 (2025-09-08)] → [Data: 145 mph]"
            },
            {
                "question": "Which company acquired TikTok after the 2025 US data privacy legislation?",
                "date": "2026-01-15",
                "answer": "Oracle",
                "causal_trace": "[DATA Act Enforcement] → [TikTok Divestment Requirement] → [Oracle Acquisition Bid] → [CFIUS Approval] → [Deal Closure (2026-01-12)]"
            },
            {
                "question": "How many member states ratified the WHO Pandemic Treaty by the implementation deadline?",
                "date": "2025-05-31",
                "answer": "137",
                "causal_trace": "[Treaty Deadline: 2025-05-30] → [Member State Ratification Tracking] → [WHO Secretariat Tally] → [Final Count: 137] → [Press Release (2025-05-28)]"
            },
            {
                "question": "What was the final altitude record for SolarStratos' sun-powered aircraft?",
                "date": "2025-11-20",
                "answer": "90,000 feet",
                "causal_trace": "[Stratospheric Test Flight] → [Telemetry Data Collection] → [Record Validation] → [Official Announcement (2025-11-18)] → [Altitude: 90,000 ft]"
            },
            {
                "question": "Did SpaceX's Starship achieve Mars orbital insertion in 2025?",
                "date": "2026-02-01",
                "answer": "Yes",
                "causal_trace": "[Starship SN-25 Launch] → [Trans-Mars Injection] → [Orbital Insertion Burn] → [NASA Deep Space Network Confirmation] → [Mission Success Notification]"
            },
            {
                "question": "What percentage of Singapore's water supply came from desalination in 2025?",
                "date": "2026-03-10",
                "answer": "65%",
                "causal_trace": "[Tuas Mega-Plant Operation] → [Annual Water Production Data] → [PUB Statistical Analysis] → [2025 Water Report] → [Desalination Contribution: 65%]"
            },
            {
                "question": "Who won the 2025 Women's World Cup final?",
                "date": "2025-08-25",
                "answer": "Spain",
                "causal_trace": "[Tournament Final Match] → [Match Result: Spain 3-1 Canada] → [FIFA Match Report] → [Trophy Presentation] → [Official FIFA Archives]"
            },
            {
                "question": "What carbon price did Canada implement under the 2025 Climate Accountability Act?",
                "date": "2025-07-01",
                "answer": "$170 per tonne",
                "causal_trace": "[Act Legislative Process] → [Royal Assent] → [Carbon Pricing Schedule] → [Regulation Text] → [Effective Rate: $170/tonne]"
            }
        ]
        
        try:
            with open(data_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
            print(f"✅ Created sample dynamodata.json with {len(sample_data)} examples")
            print(f"📝 You can replace this with your own data at: {data_file}")
        except Exception as e:
            print(f"❌ Failed to create sample data: {e}")
            return False
    
    # Validate existing file
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print("❌ Data file is empty or invalid format")
            return False
        
        # Check first sample structure
        sample = data[0]
        required_keys = ['question', 'date', 'answer', 'causal_trace']
        
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            print(f"❌ Missing required keys in data: {missing_keys}")
            print(f"Required format: {required_keys}")
            return False
        
        print(f"✅ Found {len(data)} valid training samples in dynamodata.json")
        
        # Show sample
        print(f"📄 Sample question: {sample['question'][:60]}...")
        print(f"📅 Sample date: {sample['date']}")
        print(f"💡 Sample answer: {sample['answer']}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return False

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Install basic packages
    basic_packages = [
        "torch>=2.0.0",
        "transformers>=4.21.0", 
        "scikit-learn",
        "numpy",
        "tqdm",
        "matplotlib"
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    # Install torch-geometric (can be tricky)
    print("🔧 Installing PyTorch Geometric...")
    pyg_install_cmd = "pip install torch-geometric"
    if not run_command(pyg_install_cmd, "Installing torch-geometric"):
        print("⚠️  torch-geometric installation failed, trying alternative...")
        alt_cmd = "pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html"
        if not run_command(alt_cmd, "Installing torch-geometric (alternative)"):
            print("❌ Could not install torch-geometric")
            return False
    
    # Install spaCy and model
    if not run_command("pip install spacy", "Installing spaCy"):
        return False
    
    if not run_command("python -m spacy download en_core_web_sm", "Installing spaCy model"):
        print("⚠️  spaCy model download failed, will try during runtime")
    
    # Install HuggingFace
    if not run_command("pip install huggingface-hub accelerate", "Installing HuggingFace tools"):
        return False
    
    print("✅ All dependencies installed")
    return True

def create_sample_config():
    """Create sample configuration files"""
    print("⚙️  Creating configuration files...")
    
    # Model config
    model_config = {
        "model_sizes": ["llama-7b", "llama-13b", "llama-30b"],
        "transformer": {
            "llama-7b": "meta-llama/Llama-2-7b-hf",
            "llama-13b": "meta-llama/Llama-2-13b-hf",
            "llama-30b": "meta-llama/Llama-2-30b-hf"
        },
        "batch_size": {
            "llama-7b": 4,
            "llama-13b": 2,
            "llama-30b": 1
        },
        "learning_rate": {
            "llama-7b": 5e-5,
            "llama-13b": 3e-5,
            "llama-30b": 1e-5
        },
        "time2vec_dim": 64,
        "num_nodes": 100,
        "node_dim": 128,
        "gnn_output_dim": 64,
        "fused_dim": 256,
        "num_classes": 10,
        "epochs": 3,
        "use_time2vec": True,
        "use_gnn": True,
        "use_causal_reg": True,
        "lambda_causal": 0.1,
        "time_delta": 86400
    }
    
    with open("model/configs.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("✅ Configuration files created")
    return True

def check_huggingface_login():
    """Check HuggingFace login status"""
    print("🤗 Checking HuggingFace login...")
    
    try:
        result = subprocess.run("huggingface-cli whoami", shell=True, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✅ Logged in as: {username}")
            return True
        else:
            print("⚠️  Not logged into HuggingFace")
            print("   For LLaMA models, run: huggingface-cli login")
            print("   Or use alternative models like GPT-2")
            return False
    except Exception as e:
        print("⚠️  Could not check HuggingFace status")
        return False

def run_quick_test():
    """Run a quick system test"""
    print("🧪 Running quick system test...")
    
    test_script = """
import torch
import json
import sys

# Test 1: PyTorch
print("Testing PyTorch...")
x = torch.randn(10, 10)
if torch.cuda.is_available():
    x = x.cuda()
    print("✅ PyTorch + CUDA working")
else:
    print("✅ PyTorch working (CPU only)")

# Test 2: Transformers
print("Testing Transformers...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer("Hello world")
    print("✅ Transformers working")
except Exception as e:
    print(f"❌ Transformers test failed: {e}")
    sys.exit(1)

# Test 3: PyTorch Geometric
print("Testing PyTorch Geometric...")
try:
    import torch_geometric
    print("✅ PyTorch Geometric working")
except Exception as e:
    print(f"❌ PyTorch Geometric test failed: {e}")
    sys.exit(1)

print("✅ All tests passed!")
"""
    
    # Write and run test
    with open("system_test.py", 'w') as f:
        f.write(test_script)
    
    success = run_command("python system_test.py", "Running system test")
    
    # Cleanup
    if os.path.exists("system_test.py"):
        os.remove("system_test.py")
    
    return success

def main():
    print("🚀 DYNAMO Setup")
    print("=" * 50)
    
    success_count = 0
    total_checks = 8
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", create_directory_structure),
        ("Data File", verify_data_file),
        ("Dependencies", install_dependencies),
        ("Configuration", create_sample_config),
        ("GPU Check", check_gpu),
        ("HuggingFace Login", check_huggingface_login),
        ("System Test", run_quick_test)
    ]
    
    for check_name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"CHECK: {check_name}")
        print(f"{'='*50}")
        
        if check_func():
            success_count += 1
        else:
            print(f"❌ {check_name} failed")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📋 SETUP SUMMARY")
    print(f"{'='*60}")
    
    print(f"Completed: {success_count}/{total_checks} checks")
    
    if success_count == total_checks:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\n🚀 Ready to run DYNAMO!")
        print("Next steps:")
        print("  1. python run_pipeline.py --quick_test")
        print("  2. python run_pipeline.py (for full training)")
    elif success_count >= total_checks - 2:
        print("⚠️  SETUP MOSTLY COMPLETE")
        print("You can proceed but may encounter some issues.")
        print("Try: python run_pipeline.py --quick_test")
    else:
        print("❌ SETUP FAILED")
        print("Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
