# Tempiril overview
yada yada take from paper

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

# TEMPiRL Running

TEMPiRL is a parameter-efficient framework that enhances LLMs with efficient temporal and causal reasoning capabilities for time-sensitive question answering, avoiding full retraining.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (A100 recommended, RTX 3090/4090 minimum)
- Hugging Face access token for LLaMA-2 models
- At least 16GB GPU memory for 7B model

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
## Quick Start 



### Step 1: Preprocess Data
Convert raw JSON to model-ready format:

```bash
# This creates processed_dynamodata_qa.pt from dynamodata.json
python model/processing.py
```

**Expected output**: `processed_dynamodata_qa.pt` file with tokenized data and causal graphs.

### Step 2: Train  Model
```bash
# Full training (2-3 hours on A100, 6-8 hours on RTX 4090)
python model/train.py \
    --data_path processed_dynamodata_qa.pt \
    --transformer meta-llama/Llama-2-7b-hf \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --device cuda
```

**For quick testing** (reduces training time to ~30 minutes):
```bash
# Minimal training for testing pipeline
python model/train.py \
    --data_path processed_dynamodata_qa.pt \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --device cuda
```

**Expected output**: 
- Training loss decreasing from ~10 to ~2-3
- Model saved as `dynamo_qa_model_final.pth`
- Checkpoints saved during training

### Step 4: Test Inference
```bash
# Note: You need the node_list from preprocessing
python model/inference.py \
    --model_path dynamo_qa_model_final.pth \
    --question "What was the peak sustained wind speed of Hurricane Felix at landfall in Florida?" \
    --date "2025-09-10" \
    --causal_trace "[NHC Landfall Advisory] → [Category 4 Classification] → [Peak Wind Measurement Protocol] → [Advisory #24 (2025-09-08)] → [Data: 145 mph]" \
    --device cuda
```

**Expected output**: Generated answer like "145 mph" based on the temporal and causal context.

## Data Format

Each training example in `data/dynamodata.json` contains:
```json
{
  "question": "What was the peak wind speed...",
  "date": "2025-09-10", 
  "answer": "145 mph",
  "causal_trace": "[Source] → [Process] → [Result]"
}
```

The causal trace shows the reasoning chain the model should follow.

## Troubleshooting

**Common Issues:**

1. **CUDA OOM Error**: 
   ```bash
   # Reduce batch size
   python model/train.py --batch_size 1
   
   # Or use gradient accumulation
   python model/train.py --batch_size 1 --gradient_accumulation_steps 4
   ```

2. **HuggingFace Access Denied**: 
   ```bash
   # Check access
   huggingface-cli whoami
   
   # Request access to LLaMA-2 on HuggingFace Hub first
   ```

3. **spaCy Model Missing**: 
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Preprocessing Fails**: 
   - Ensure `data/dynamodata.json` exists
   - Check JSON format is valid
   - Verify spaCy model is installed

5. **Training Crashes**:
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Try smaller model or batch size
   python model/train.py --batch_size 1 --gradient_accumulation_steps 2
   ```

6. **Inference Errors**:
   - Ensure model file exists: `dynamo_qa_model_final.pth`
   - Check that preprocessing created the node list correctly
   - Verify the causal trace format matches training data

## Verification Steps

To ensure everything is working correctly:

1. **After Installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"  # Should print True
   python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK')"
   ```

2. **After Preprocessing**:
   ```bash
   ls -la processed_dynamodata_qa.pt  # File should exist and be >1MB
   python -c "import torch; data = torch.load('processed_dynamodata_qa.pt'); print(f'Loaded {len(data[\"processed_data\"])} examples')"
   ```

3. **After Training**:
   ```bash
   ls -la dynamo_qa_model_final.pth  # Should be ~13GB for 7B model
   ```

## Advanced Usage

### Custom Configuration
```bash
# Copy and edit config
cp model/configs.json my_config.json
# Modify learning rates, dimensions, etc.
python model/train.py --config my_config.json
```

### Ablation Studies
```bash
# Test individual components
python model/train.py --use_time2vec false --use_gnn true    # No temporal
python model/train.py --use_time2vec true --use_gnn false   # No causal
python model/train.py --use_time2vec false --use_gnn false  # Baseline
```

### Full Evaluation
```bash
# Compare against all baselines (requires training them first)
python evals/evaluate_dynamo.py \
    --data_path data/dynamodata.json \
    --config model/configs.json \
    --dynamo_weights weights/dynamo.pt \
    --rag_weights weights/rag.pt \
    --full_ft_weights weights/full_ft.pt \
    --device cuda
```

## Expected Results

With proper setup:
- **Training**: Loss ~10 → ~2-3 over 3 epochs
- **Memory**: ~14GB GPU memory for 7B model  
- **Time**: ~3 hours training on A100, ~8 hours on RTX 4090
- **Accuracy**: 85-90% on temporal Q&A (vs ~75% for baselines)
- **Inference**: ~2-3 seconds per question

## File Structure
```
TEMPiRL/
├── model/                    # Core DYNAMO implementation
│   ├── model.py             # Main DYNAMO architecture
│   ├── train.py             # Training script
│   ├── inference.py         # Inference script  
│   └── processing.py        # Data preprocessing
├── data/
│   └── dynamodata.json      # Training data
├── causal_graph_extraction/
│   └── causal_extractor.py  # Improved causal extraction
├── evals/                   # Evaluation and baselines
└── requirements.txt
```

## Getting Help

**Debug checklist:**
1. ✅ GPU available: `nvidia-smi`
2. ✅ CUDA working: `python -c "import torch; print(torch.cuda.is_available())"`  
3. ✅ HuggingFace access: `huggingface-cli whoami`
4. ✅ Data exists: `ls data/dynamodata.json`
5. ✅ spaCy model: `python -m spacy download en_core_web_sm`

**Common fixes:**
- Reduce batch size for memory issues
- Check file paths are correct
- Ensure all dependencies installed
- Verify LLaMA-2 access approved on HuggingFace

The system should work out-of-the-box once dependencies are properly installed.




