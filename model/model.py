import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig
from torch_geometric.nn import GCNConv

class Time2Vec(nn.Module):
    """Time2Vec layer for encoding temporal information with sinusoidal functions."""
    def __init__(self, dim: int):
        """
        Args:
            dim (int): Output dimension of the Time2Vec embedding.
        """
        super().__init__()
        if dim % 2 == 0:
            self.weights = nn.Parameter(torch.randn(dim // 2 + 1))  # Linear + sin/cos pairs
            self.phases = nn.Parameter(torch.randn(dim // 2 + 1))
            self.dim = dim + 1  # Adjust for odd output due to linear term
        else:
            self.weights = nn.Parameter(torch.randn((dim + 1) // 2))
            self.phases = nn.Parameter(torch.randn((dim + 1) // 2))
            self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Input time tensor of shape [batch_size].

        Returns:
            torch.Tensor: Time embeddings of shape [batch_size, dim].
        """
        t = t.unsqueeze(-1)  # [batch_size, 1]
        linear = self.weights[0] * t + self.phases[0]  # Linear term
        sin_part = torch.sin(t * self.weights[1:] + self.phases[1:])  # Sinusoidal terms
        return torch.cat([linear, sin_part], dim=-1)[:, :self.dim]  # Adjust for exact dim

class DynamoModel(nn.Module):
    """DYNAMO model integrating LLaMA-7B, Time2Vec, and GNN for temporal-causal reasoning."""
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary with model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.use_time2vec = config.get('use_time2vec', True)
        self.use_gnn = config.get('use_gnn', True)

        # LLaMA-7B backbone
        self.transformer = LlamaModel.from_pretrained(config['transformer'])
        self.hidden_dim = self.transformer.config.hidden_size  # 4096 for LLaMA-7B

        # Time2Vec component
        self.time2vec_dim = config['time2vec_dim'] if self.use_time2vec else 0
        self.time2vec = Time2Vec(self.time2vec_dim) if self.use_time2vec else nn.Identity()

        # GNN component for causal graph processing
        self.node_dim = config['node_dim'] if self.use_gnn else 0
        self.gnn_output_dim = config['gnn_output_dim'] if self.use_gnn else 0
        if self.use_gnn:
            self.node_embedding = nn.Embedding(config['num_nodes'], self.node_dim)
            self.gnn = GCNConv(self.node_dim, self.gnn_output_dim)
        else:
            self.node_embedding = None
            self.gnn = None

        # Adapter for temporal and causal fusion
        input_dim = self.hidden_dim + self.time2vec_dim + self.gnn_output_dim
        self.fusion_layer = nn.Linear(input_dim, config['fused_dim'])
        self.output_head = nn.Linear(config['fused_dim'], config['num_classes'])

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                time: torch.Tensor,
                edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): Tokenized input of shape [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len].
            time (torch.Tensor): Time values of shape [batch_size].
            edge_indices (torch.Tensor): Graph edge indices of shape [2, num_edges].

        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes].
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Text embedding from LLaMA-7B (using last hidden state of first token)
        text_emb = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]  # [batch_size, hidden_dim]

        # Time embedding
        time_emb = self.time2vec(time) if self.use_time2vec else torch.zeros(batch_size, self.time2vec_dim, device=device)

        # Graph embedding
        if self.use_gnn:
            node_emb = self.node_embedding.weight  # [num_nodes, node_dim]
            gnn_output = self.gnn(node_emb, edge_indices)  # [num_nodes, gnn_output_dim]
            graph_emb = gnn_output.mean(dim=0)  # Graph-level embedding [gnn_output_dim]
            graph_emb = graph_emb.expand(batch_size, -1)  # [batch_size, gnn_output_dim]
        else:
            graph_emb = torch.zeros(batch_size, self.gnn_output_dim, device=device)

        # Combine representations via adapter
        combined = torch.cat([text_emb, time_emb, graph_emb], dim=-1)  # [batch_size, total_dim]
        fused = torch.relu(self.fusion_layer(combined))  # [batch_size, fused_dim]
        logits = self.output_head(fused)  # [batch_size, num_classes]
        return logits
