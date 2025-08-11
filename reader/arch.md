# DYNAMO Model Architecture

DYNAMO (Dynamic Causal-Temporal Adaptation in Transformer Architectures via Time2Vec Embeddings and Graph-Conditioned Attention) is a parameter-efficient framework designed to address temporal bias and causal fragility in large language models (LLMs). Built on top of a LLaMA backbone (e.g., LLaMA-2-7B), DYNAMO introduces lightweight adapters that integrate continuous temporal embeddings and causal graph projections, enabling dynamic adaptation with only 0.11% trainable parameters.

## Key Components

### 1. LLaMA Backbone

- The core transformer model (e.g., LLaMA-2-7B with 4096 hidden dimensions) processes tokenized input text to generate base representations.
- **Input**: Tokenized sequences (input_ids, attention_mask).
- **Output**: Hidden states from the last layer (e.g., [batch_size, seq_len, hidden_dim]).

### 2. Time2Vec Embeddings

- Encodes time (t) as a continuous vector using a learnable combination of linear and sinusoidal functions:

$$\phi(t)[k] = \begin{cases} 
\omega_k t + \varphi_k & k=0 \\ 
\sin(\omega_k t + \varphi_k) & 1 \leq k \leq \lfloor d/2 \rfloor \\ 
\cos(\omega_k t + \varphi_k) & \text{otherwise} 
\end{cases}$$

- **Dimension**: Typically 64.
- **Purpose**: Captures monotonic and periodic temporal patterns, reducing temporal bias.

### 3. Causal Graph Projections (GNN)

- Processes time-varying causal graphs $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$ using a Graph Convolutional Network (GCN).
- **Input**: Edge indices (edge_indices, shape [2, num_edges]) and node embeddings (e.g., 128 dimensions).
- **Output**: Graph-level embedding (mean pooled, shape [gnn_output_dim], typically 64).
- **Adjacency matrix**: $\mathbf{A}_t[i,j] = w_{ij}^t \cdot \mathbb{I}[t \in \tau_{ij}]$

### 4. Temporal-Causal Adapters

- Modulates transformer hidden states:

$$\Delta \mathbf{H}^\ell = \mathbf{W}_o \cdot \sigma(\mathbf{W}_t \phi(t) + \mathbf{W}_g f_g(\mathcal{G}_t))$$

$$\mathbf{H}_{\text{out}}^\ell = \mathbf{H}^\ell + \Delta \mathbf{H}^\ell$$

- **Activation**: ReLU (or GELU as in paper).
- **Loss term**: Added during training with weight $\lambda = 0.1$.

## Overall Flow

- **Input**: Tokenized text, time stamp, causal graph edges.
- **Processing**:
  1. Extract text embeddings from LLaMA.
  2. Compute Time2Vec embedding.
  3. Process causal graph with GCN.
  4. Fuse via adapter and output logits.
- **Output**: Classification logits (e.g., for QA tasks).

## Ablation Variants

- **No Time2Vec**: Sets `use_time2vec = false`, zeroing time embeddings.
- **No GNN**: Sets `use_gnn = false`, zeroing graph embeddings.
- **No Causal Reg.**: Sets `use_causal_reg = false`, skipping regularization.

This architecture ensures efficiency and robustness, as validated in Test 2 (84.0% accuracy, 4.9 GPU hr for 7B model).

For code details, see `model.py`.
