import torch
import json
from metrics import compute_accuracy, compute_drift  # Assume from evals/metrics.py
from train_replay import train_model_with_replay
from train_merge import train_model_with_merging
from train import train_model  # Original DYNAMO

# Dummy data simulation for Wikipedia (3 batches, 10 samples each)
dummy_data = []
for batch in range(3):
    for i in range(10):
        input_ids = torch.randint(0, 50256, (512,))
        attention_mask = torch.ones(512)
        time = 1719792000 + batch * 7776000 + i * 86400  # Incremental timestamps
        label = torch.tensor(random.randint(0, 9))
        edge_indices = torch.tensor([[random.randint(0, 99) for _ in range(5)], [random.randint(0, 99) for _ in range(5)]])
        dummy_data.append({
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "time": time,
            "label": label.item(),
            "edge_indices": edge_indices.tolist()
        })

with open('dummy_dynamodata.json', 'w') as f:
    json.dump(dummy_data, f)

config = {  # Simplified config
    "transformer": "meta-llama/Llama-2-7b-hf",
    "batch_size": 4,
    "learning_rate": 1e-5,
    "epochs": 2,
    "use_time2vec": True,
    "use_gnn": True,
    "use_causal_reg": True,
    "lambda_causal": 0.1,
    "time_delta": 86400,
    "time2vec_dim": 64,
    "num_nodes": 100,
    "node_dim": 128,
    "gnn_output_dim": 64,
    "fused_dim": 256,
    "num_classes": 10
}



# Compute metrics (use compute_accuracy/drift from metrics.py; simulate here)
results = {
    "dynamo": {"accuracy": dynamo_accuracy, "time_hr": dynamo_time, "drift": dynamo_drift, "delta_gain": dynamo_delta_gain},
    "replay_buffers": {"accuracy": replay_accuracy, "time_hr": replay_time, "drift": replay_drift, "delta_gain": replay_delta_gain},
    "model_merging": {"accuracy": merge_accuracy, "time_hr": merge_time, "drift": merge_drift, "delta_gain": merge_delta_gain}
}

with open('comparison_results.json', 'w') as f:
    json.dump(results, f)

print(results)
