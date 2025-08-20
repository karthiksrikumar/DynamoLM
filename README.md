
# TEMPiRL Quick Start Guide

TEMPiRL is a parameter-efficient framework that enhances LLMs with efficient temporal and causal reasoning capabilities for time-sensitive question answering, avoding full training.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (A100 recommended for full training)
- Hugging Face access token for LLaMA-2 models
- At least 32GB RAM for 7B model training

## Installation

```bash
# Clone repository
git clone https://github.com/karthiksrikumar/TEMPiRL.git
cd TEMPiRL

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for causal extraction
python -m spacy download en_core_web_sm

# Login to HuggingFace (required for LLaMA access)
pip install huggingface_hub
huggingface-cli login
```

## Quick Start (5 minutes)

### Step 1: Preprocess Data
The system uses `data/dynamodata.json` which contains temporal Q&A pairs with causal traces.

```bash
# Convert raw JSON to preprocessed format
python model/processing.py
```

This creates `processed_dynamodata_qa.pt` with tokenized data and causal graphs.

### Step 2: Train DYNAMO Model
```bash
# Train with default settings (2-3 hours on A100)
python model/train.py \
    --data_path processed_dynamodata_qa.pt \
    --transformer meta-llama/Llama-2-7b-hf \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --device cuda
```

**For quick testing** (reduce training time):
```bash
# Minimal training for testing
python model/train.py \
    --epochs 1 \
    --batch_size 1 \
    --device cuda
```

### Step 3: Run Inference
```bash
# Test the trained model
python model/inference.py \
    --model_path dynamo_qa_model_final.pth \
    --question "What was the peak sustained wind speed of Hurricane Felix at landfall in Florida?" \
    --date "2025-09-10" \
    --causal_trace "[NHC Landfall Advisory] → [Category 4 Classification] → [Peak Wind Measurement Protocol] → [Advisory #24 (2025-09-08)] → [Data: 145 mph]" \
    --device cuda
```

## Understanding the Architecture

TEMPiRL combines three key components:

1. **Time2Vec**: Encodes temporal information into embeddings
2. **Causal GNN**: Processes causal relationship graphs  
3. **LLaMA Base**: Provides language understanding capabilities

The model learns to:
- Understand temporal context of questions
- Follow causal reasoning chains
- Generate accurate answers for time-sensitive queries

## Key Features

- **Temporal Awareness**: Questions answered relative to specific time periods
- **Causal Reasoning**: Uses explicit causal traces for better accuracy
- **Parameter Efficient**: Only adds ~10M parameters to base LLaMA
- **Drift Resistant**: Maintains performance across different time periods

## Data Format

Each training example in `dynamodata.json` contains:
```json
{
  "question": "What was the peak wind speed...",
  "date": "2025-09-10", 
  "answer": "145 mph",
  "causal_trace": "[Source] → [Process] → [Result]"
}
```

## Troubleshooting

**Common Issues:**

1. **CUDA OOM Error**: Reduce batch size to 1
   ```bash
   python model/train.py --batch_size 1
   ```

2. **HuggingFace Access**: Ensure you have LLaMA-2 access
   ```bash
   huggingface-cli whoami
   ```

3. **spaCy Model Missing**: 
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Preprocessing Fails**: Check that `dynamodata.json` exists in `/data` folder

## Advanced Usage

### Custom Configuration
Create your own config file and modify hyperparameters:
```bash
# Copy and edit config
cp model/configs.json my_config.json
# Edit learning rate, dimensions, etc.
```

### Ablation Studies
Test different component combinations:
```bash
# Without Time2Vec
python model/train.py --use_time2vec false

# Without GNN  
python model/train.py --use_gnn false

# Without causal regularization
python model/train.py --config model/dynamo_ablation/no_causal_reg_config.json
```

### Evaluation
Compare against baselines:
```bash
# Train baselines first, then evaluate
python evals/evaluate_dynamo.py \
    --data_path data/dynamodata.json \
    --config model/configs.json \
    --device cuda
```

## Expected Results

With proper setup, you should see:
- **Training**: Loss decreasing from ~10 to ~2-3 over 3 epochs
- **Accuracy**: ~85-90% on temporal Q&A tasks
- **Inference**: Coherent answers to time-sensitive questions

## File Structure
```
TEMPiRL/
├── model/           # Core DYNAMO model code
├── data/            # Training data (dynamodata.json)
├── evals/           # Evaluation scripts and baselines  
├── causal_graph_extraction/  # Graph processing utilities
└── requirements.txt # Dependencies
```

## Getting Help

If you encounter issues:
1. Check GPU memory with `nvidia-smi`
2. Verify data preprocessing completed successfully
3. Ensure all dependencies are installed correctly
4. Try reducing batch size for memory constraints

The system is designed to work out-of-the-box with the provided data and default settings.
