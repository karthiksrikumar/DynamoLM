import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig
from torch_geometric.nn import GCNConv
from typing import Dict, Optional, Tuple, Union
import warnings


class Time2Vec(nn.Module):
    
    def __init__(self, dim: int, activation: str = "sin"):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
            
        self.dim = dim
        self.activation = activation
        
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.randn(1))
        
        if dim > 1:
            self.periodic_weights = nn.Parameter(torch.randn(dim - 1))
            self.periodic_biases = nn.Parameter(torch.randn(dim - 1))
        else:
            self.register_parameter('periodic_weights', None)
            self.register_parameter('periodic_biases', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_weight.unsqueeze(0))
        nn.init.zeros_(self.linear_bias)
        
        if self.periodic_weights is not None:
            nn.init.xavier_uniform_(self.periodic_weights.unsqueeze(0))
            nn.init.uniform_(self.periodic_biases, -torch.pi, torch.pi)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.numel() == 0:
            return torch.empty(*t.shape, self.dim, device=t.device, dtype=t.dtype)
            
        original_shape = t.shape
        t_flat = t.flatten().unsqueeze(-1)
        
        linear_component = self.linear_weight * t_flat + self.linear_bias
        
        if self.dim == 1:
            result = linear_component
        else:
            periodic_input = self.periodic_weights * t_flat + self.periodic_biases
            
            if self.activation == "sin":
                periodic_component = torch.sin(periodic_input)
            elif self.activation == "cos":
                periodic_component = torch.cos(periodic_input)
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
            
            result = torch.cat([linear_component, periodic_component], dim=-1)
        
        return result.view(*original_shape, self.dim)


class GraphPooling(nn.Module):
    
    def __init__(self, pooling_type: str = "mean"):
        super().__init__()
        self.pooling_type = pooling_type
        
        if pooling_type not in ["mean", "max", "sum", "attention"]:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
    
    def forward(self, node_embeddings: torch.Tensor, batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pooling_type == "mean":
            return node_embeddings.mean(dim=0, keepdim=True)
        elif self.pooling_type == "max":
            return node_embeddings.max(dim=0, keepdim=True)[0]
        elif self.pooling_type == "sum":
            return node_embeddings.sum(dim=0, keepdim=True)
        elif self.pooling_type == "attention":
            attention_weights = torch.softmax(node_embeddings.sum(dim=-1), dim=0)
            return (attention_weights.unsqueeze(-1) * node_embeddings).sum(dim=0, keepdim=True)


class DynamoModel(nn.Module):
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self._validate_config()
        
        self.use_time2vec = config.get('use_time2vec', True)
        self.use_gnn = config.get('use_gnn', True)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.freeze_transformer = config.get('freeze_transformer', False)
        
        self._init_transformer()
        self._init_time2vec()
        self._init_gnn()
        self._init_fusion_layers()
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def _validate_config(self):
        required_keys = ['transformer', 'num_classes']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config['num_classes'] <= 0:
            raise ValueError("num_classes must be positive")
    
    def _init_transformer(self):
        try:
            self.transformer = LlamaModel.from_pretrained(
                self.config['transformer'],
                torch_dtype=torch.float32,
                device_map=None
            )
            self.hidden_dim = self.transformer.config.hidden_size
            
            if self.freeze_transformer:
                for param in self.transformer.parameters():
                    param.requires_grad = False
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer model: {e}")
    
    def _init_time2vec(self):
        if self.use_time2vec:
            self.time2vec_dim = self.config.get('time2vec_dim', 128)
            self.time2vec = Time2Vec(self.time2vec_dim)
        else:
            self.time2vec_dim = 0
            self.time2vec = None
    
    def _init_gnn(self):
        if self.use_gnn:
            self.node_dim = self.config.get('node_dim', 256)
            self.gnn_output_dim = self.config.get('gnn_output_dim', 256)
            self.gnn_layers = self.config.get('gnn_layers', 2)
            self.pooling_type = self.config.get('pooling_type', 'mean')
            
            self.node_embedding = None
            
            self.gnn_convs = nn.ModuleList()
            for i in range(self.gnn_layers):
                in_dim = self.node_dim if i == 0 else self.gnn_output_dim
                self.gnn_convs.append(GCNConv(in_dim, self.gnn_output_dim))
            
            self.graph_pooling = GraphPooling(self.pooling_type)
            self.gnn_norm = nn.LayerNorm(self.gnn_output_dim)
        else:
            self.gnn_output_dim = 0
            self.gnn_convs = None
            self.graph_pooling = None
            self.node_embedding = None
    
    def _init_fusion_layers(self):
        input_dim = self.hidden_dim + self.time2vec_dim + self.gnn_output_dim
        
        if input_dim == 0:
            raise ValueError("At least one component must be enabled")
        
        self.fused_dim = self.config.get('fused_dim', 512)
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(input_dim, self.fused_dim),
            nn.LayerNorm(self.fused_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fused_dim, self.fused_dim),
            nn.LayerNorm(self.fused_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.output_head = nn.Linear(self.fused_dim, self.config['num_classes'])
    
    def _initialize_node_embeddings(self, num_nodes: int, device: torch.device):
        if self.use_gnn and (self.node_embedding is None or self.node_embedding.num_embeddings != num_nodes):
            self.node_embedding = nn.Embedding(num_nodes, self.node_dim).to(device)
            nn.init.xavier_uniform_(self.node_embedding.weight)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                time: Optional[torch.Tensor] = None,
                edge_indices: Optional[torch.Tensor] = None,
                node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        embeddings = []
        
        text_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self._pool_text_embeddings(text_outputs.last_hidden_state, attention_mask)
        embeddings.append(text_emb)
        
        if self.use_time2vec and time is not None:
            time_emb = self.time2vec(time)
            embeddings.append(time_emb)
        elif self.use_time2vec:
            time_emb = self.time2vec(torch.zeros(batch_size, device=device))
            embeddings.append(time_emb)
        
        if self.use_gnn and edge_indices is not None:
            graph_emb = self._process_graph(edge_indices, batch_size, device)
            embeddings.append(graph_emb)
        elif self.use_gnn:
            graph_emb = torch.zeros(batch_size, self.gnn_output_dim, device=device)
            embeddings.append(graph_emb)
        
        if len(embeddings) == 0:
            raise RuntimeError("No valid embeddings computed")
        
        combined = torch.cat(embeddings, dim=-1)
        combined = self.dropout(combined)
        
        fused = self.fusion_layers(combined)
        logits = self.output_head(fused)
        
        return logits
    
    def _pool_text_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        
        masked_embeddings = hidden_states * attention_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)
        
        mean_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
        return mean_embeddings
    
    def _process_graph(self, edge_indices: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if edge_indices.numel() == 0:
            return torch.zeros(batch_size, self.gnn_output_dim, device=device)
        
        num_nodes = edge_indices.max().item() + 1
        self._initialize_node_embeddings(num_nodes, device)
        
        node_emb = self.node_embedding.weight
        
        x = node_emb
        for conv in self.gnn_convs:
            x = conv(x, edge_indices)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.gnn_norm(x)
        
        graph_emb = self.graph_pooling(x)
        graph_emb = graph_emb.expand(batch_size, -1)
        
        return graph_emb
    
    def get_config(self) -> Dict:
        return self.config.copy()
    
    def count_parameters(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_dynamo_model(transformer_path: str, num_classes: int, **kwargs) -> DynamoModel:
    default_config = {
        'transformer': transformer_path,
        'num_classes': num_classes,
        'use_time2vec': True,
        'use_gnn': True,
        'time2vec_dim': 128,
        'node_dim': 256,
        'gnn_output_dim': 256,
        'gnn_layers': 2,
        'fused_dim': 512,
        'dropout_rate': 0.1,
        'pooling_type': 'mean',
        'freeze_transformer': False
    }
    
    default_config.update(kwargs)
    
    return DynamoModel(default_config)


if __name__ == "__main__":
    config = {
        'transformer': 'meta-llama/Llama-2-7b-hf',
        'num_classes': 3,
        'use_time2vec': True,
        'use_gnn': True,
        'time2vec_dim': 64,
        'node_dim': 128,
        'gnn_output_dim': 128,
        'fused_dim': 256,
        'dropout_rate': 0.1
    }
    
    model = DynamoModel(config)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # batch_size, seq_len = 2, 128
    # input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    # attention_mask = torch.ones(batch_size, seq_len)
    # time = torch.randn(batch_size)
    # edge_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
    #
    # with torch.no_grad():
    #     logits = model(input_ids, attention_mask, time, edge_indices)
    #     print(f"Output shape: {logits.shape}")
