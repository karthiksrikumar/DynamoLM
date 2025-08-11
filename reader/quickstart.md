# Quickstart: Using the DYNAMO Repo in <10 Minutes

This guide gets you up and running with DYNAMO for temporal-causal adaptation on LLaMA models.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (A100 recommended for Test 2)
- Hugging Face access for LLaMA-2 models (request at meta.com)

## Step 1: Clone and Install (1 min)

```bash
git clone https://github.com/your-repo/dynamo.git
cd dynamo
pip install -r requirements.txt  # Installs torch, transformers, torch-geometric
python -m spacy download en_core_web_sm  # For causal extraction
```

## Step 2: Prepare Data (2 min)

- Use `data/dynamodata.json` template or generate from Wikipedia:

```bash
python preprocess.py --wiki_dir data/wikipedia --output data/dynamodata.json
```

- **Format**: See `data_format.md`.

## Step 3: Train DYNAMO (3 min)

Train on Test 2 dataset:

```bash
python train.py --variant full --data_path data/dynamodata.json --device cuda
```

- Trains LLaMA-2-7B + DYNAMO for 10 epochs (~4.9 GPU hr on A100).
- Saves weights to `weights/dynamo.pt`.

## Step 4: Evaluate (2 min)

Run evaluation on Test 2:

```bash
python evals/evaluate_dynamo.py --data_path data/dynamodata.json --config config.json \
    --dynamo_weights weights/dynamo.pt --rag_weights weights/rag.pt \
    --full_ft_weights weights/full_ft.pt --output_path evals/results.json --device cuda
```

- Outputs `evals/results.json` with metrics (e.g., 84.0% accuracy, 5.0% drift).
- Visualize:

```bash
python evals/plots/accuracy_vs_time.py --results_path evals/results.json
```

## Step 5: Run Ablations (2 min)

```bash
python evals/ablation_runner.py --data_path data/dynamodata.json --device cuda
```

- Generates `evals/ablation_results/full.json`, etc.

## Troubleshooting

- **OOM errors**: Reduce batch size in `config.json`.
- **Data issues**: Verify timestamps and edges in `dynamodata.json`.
- **Scaling**: Edit `config.json` for 13B/30B and rerun.

**Total time**: ~10 min. For details, see `architecture.md`.
