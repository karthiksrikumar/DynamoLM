# DYNAMO: Dynamic Temporal-Causal Reasoning for Question Answering

## Abstract

DYNAMO is a novel neural architecture that integrates temporal embeddings and causal graph representations with large language models for improved question answering on time-sensitive, causally-structured data. This repository contains the complete implementation, evaluation framework, and experimental protocols for reproducing the results presented in our research paper.

## Architecture Overview

DYNAMO extends transformer-based language models with two key components:
- **Time2Vec temporal embeddings**: Capture temporal patterns and dependencies
- **Graph Neural Networks (GNNs)**: Model explicit causal relationships between entities

The architecture fuses these modalities through learned projection layers, enabling the model to reason about temporal causality in question-answering tasks.

## Repository Structure

```
TEMPiRL/
├── model/                           # Core DYNAMO implementation
│   ├── model.py                    # Main DYNAMO architecture
│   ├── fullpipeline.py             # Complete training pipeline
│   ├── train.py                    # Training utilities and DynamoTrainer
│   ├── inference.py                # Inference and generation
│   ├── tokenizer_utils.py          # Text processing utilities
│   ├── temporal_data_splitter.py   # Temporal data splitting (prevents leakage)
│   ├── reproducibility.py          # Reproducibility controls
│   ├── data_validator.py           # Data validation framework
│   └── configs.json                # Model hyperparameters
├── data/
│   └── dynamodata.json             # Temporal Q&A dataset
├── evals/                          # Evaluation framework
│   ├── evaluate_dynamo.py          # Main evaluation script
│   ├── metrics.py                  # Evaluation metrics
│   └── models/                     # Baseline implementations
│       ├── compRAG.py              # Retrieval-Augmented Generation baseline
│       └── full_finetune.py        # Full fine-tuning baseline
├── causal_graph_extraction/        # Causal graph processing
│   ├── causal_extractor.py         # Causal relationship extraction
│   ├── entity_extractor.py         # Named entity recognition
│   ├── temporal_graph.py           # Temporal graph construction
│   └── evals/                      # Causal extraction evaluation
└── requirements.txt                # Dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)
- 16GB+ GPU memory for LLaMA-7B (8GB for CPU training)
- HuggingFace account with LLaMA-2 access

### Setup

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd TEMPiRL
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install spaCy language model:**
```bash
python -m spacy download en_core_web_sm
```

3. **Configure HuggingFace access:**
```bash
huggingface-cli login
# Ensure access to meta-llama/Llama-2-7b-hf
```

## Usage

### Quick Start

**Complete pipeline with data leakage protection:**
```bash
python model/fullpipeline.py --quick_test
```

**Full training (3-8 hours depending on hardware):**
```bash
python model/fullpipeline.py \
    --data data/dynamodata.json \
    --transformer meta-llama/Llama-2-7b-hf \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --device cuda
```

### Evaluation

**Comparative evaluation against baselines:**
```bash
python evals/evaluate_dynamo.py \
    --data_path data/dynamodata.json \
    --config model/configs.json \
    --dynamo_weights checkpoints/best_model.pt \
    --rag_weights weights/rag.pt \
    --full_ft_weights weights/full_ft.pt \
    --device cuda
```

### Inference

**Single question inference:**
```bash
python model/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --mode single \
    --question "What was the peak wind speed of Hurricane Felix?" \
    --date "2025-09-10" \
    --causal_trace "[NHC Advisory] → [Category 4] → [Peak Wind Measurement] → [145 mph]"
```

**Batch inference:**
```bash
python model/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --mode batch \
    --input_file test_questions.json \
    --output_file results.json
```

## Experimental Design

### Temporal Data Splitting

To prevent data leakage, we implement strict temporal splitting:
- **Training**: January 2025 - August 2025
- **Validation**: September 2025 - December 2025  
- **Test**: January 2026 - December 2026

This ensures the model never sees future information during training, maintaining temporal validity.

**⚠️ CRITICAL:** Evaluation scripts now use the same temporal ranges as training to prevent data leakage. Previous versions had inconsistent date ranges that could invalidate results.

### Baseline Comparisons

1. **RAG (Retrieval-Augmented Generation)**: LLaMA-7B + DPR retrieval
2. **Full Fine-tuning**: Standard LLaMA-7B fine-tuned on the same data
3. **DYNAMO**: Our proposed architecture with temporal + causal features

### Reproducibility

All experiments use deterministic training with fixed seeds:
```python
set_all_seeds(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

## Key Features

### Data Leakage Prevention
- ✅ Strict temporal splitting with validation
- ✅ No test data used during training or model selection
- ✅ Consistent preprocessing across all splits

### Reproducibility Controls
- ✅ Fixed random seeds across all components
- ✅ Deterministic algorithms enabled
- ✅ Consistent data loading and batching

### Fair Evaluation
- ✅ Each model evaluated with its designed input modalities
- ✅ DYNAMO gets temporal + graph features (research contribution)
- ✅ Baselines get their respective designed inputs

## Expected Results

With proper setup on recommended hardware:

| Metric | DYNAMO | RAG | Full Fine-tune |
|--------|--------|-----|----------------|
| Accuracy | 85-90% | ~75% | ~70% |
| Temporal Drift | <5% | 8-12% | 10-15% |
| Training Time | 3-8 hours | 2-6 hours | 2-5 hours |
| Memory Usage | 14-16GB | 12-14GB | 10-12GB |

## Hardware Requirements

### Minimum (CPU Training)
- 32GB RAM
- 50GB disk space
- Training time: 24-48 hours

### Recommended (GPU Training)
- NVIDIA GPU with 16GB+ VRAM (A100, RTX 4090, etc.)
- 32GB system RAM
- 100GB disk space
- Training time: 3-8 hours

### Cloud Alternatives
- AWS p3.2xlarge or p4d.xlarge
- Google Cloud A100 instances
- Azure NC-series VMs

## Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
# Reduce batch size
python model/fullpipeline.py --batch_size 1

# Use gradient checkpointing (add to config)
"gradient_checkpointing": true
```

**HuggingFace access denied:**
```bash
# Ensure LLaMA-2 access approved
huggingface-cli whoami
# Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf
```

**Import errors:**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Validation Commands

**Check installation:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from model.fullpipeline import process_data; print('Pipeline imports OK')"
```

**Validate data:**
```bash
python model/data_validator.py --data_path data/dynamodata.json
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dynamo2025,
  title={DYNAMO: Dynamic Temporal-Causal Reasoning for Question Answering},
  author={[Author Names]},
  journal={[Conference/Journal]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines and submit pull requests for:
- Bug fixes
- Performance improvements  
- Additional baseline implementations
- Extended evaluation metrics

## Acknowledgments

- HuggingFace for transformer models and tokenizers
- PyTorch Geometric for graph neural network implementations
- The research community for temporal reasoning benchmarks

---

**Contact**: [Contact Information]  
**Paper**: [arXiv/Conference Link]  
**Documentation**: [Additional Documentation Link]