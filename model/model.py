import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from torch_geometric.nn import GCNConv
from typing import Dict, Optional, Tuple, Union, List
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
        required_keys = ['transformer']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def _init_transformer(self):
        try:
            # Use LlamaForCausalLM instead of LlamaModel for generation
            self.transformer = LlamaForCausalLM.from_pretrained(
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
        # For QA, we'll fuse the additional features into the transformer's input embeddings
        input_dim = self.hidden_dim + self.time2vec_dim + self.gnn_output_dim
        
        if input_dim == 0:
            raise ValueError("At least one component must be enabled")
        
        self.fused_dim = self.config.get('fused_dim', 512)
        
        # Projection layer to map fused features back to the transformer's hidden size
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, self.fused_dim),
            nn.LayerNorm(self.fused_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fused_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh()
        )
    
    def _initialize_node_embeddings(self, num_nodes: int, device: torch.device):
        if self.use_gnn and (self.node_embedding is None or self.node_embedding.num_embeddings != num_nodes):
            self.node_embedding = nn.Embedding(num_nodes, self.node_dim).to(device)
            nn.init.xavier_uniform_(self.node_embedding.weight)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                time: Optional[torch.Tensor] = None,
                edge_indices: Optional[torch.Tensor] = None,
                node_indices: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Get base transformer embeddings
        base_embeddings = self.transformer.get_input_embeddings()(input_ids)
        
        # Process additional features
        additional_features = []
        
        if self.use_time2vec and time is not None:
            time_emb = self.time2vec(time)
            # Expand time embedding to match sequence length
            time_emb = time_emb.unsqueeze(1).expand(-1, base_embeddings.size(1), -1)
            additional_features.append(time_emb)
        
        if self.use_gnn and edge_indices is not None:
            graph_emb = self._process_graph(edge_indices, batch_size, device)
            # Expand graph embedding to match sequence length
            graph_emb = graph_emb.unsqueeze(1).expand(-1, base_embeddings.size(1), -1)
            additional_features.append(graph_emb)
        
        # Combine additional features if any
        if additional_features:
            combined_features = torch.cat(additional_features, dim=-1)
            # Project to the same dimension as base embeddings
            projected_features = self.feature_projection(combined_features)
            # Add to base embeddings
            inputs_embeds = base_embeddings + projected_features
        else:
            inputs_embeds = base_embeddings
        
        # Pass through transformer
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
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
    
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 time: Optional[torch.Tensor] = None,
                 edge_indices: Optional[torch.Tensor] = None,
                 node_indices: Optional[torch.Tensor] = None,
                 **generation_kwargs) -> torch.Tensor:
        """
        Generate answers using the model.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Get base transformer embeddings
        base_embeddings = self.transformer.get_input_embeddings()(input_ids)
        
        # Process additional features
        additional_features = []
        
        if self.use_time2vec and time is not None:
            time_emb = self.time2vec(time)
            time_emb = time_emb.unsqueeze(1).expand(-1, base_embeddings.size(1), -1)
            additional_features.append(time_emb)
        
        if self.use_gnn and edge_indices is not None:
            graph_emb = self._process_graph(edge_indices, batch_size, device)
            graph_emb = graph_emb.unsqueeze(1).expand(-1, base_embeddings.size(1), -1)
            additional_features.append(graph_emb)
        
        # Combine additional features if any
        if additional_features:
            combined_features = torch.cat(additional_features, dim=-1)
            projected_features = self.feature_projection(combined_features)
            inputs_embeds = base_embeddings + projected_features
        else:
            inputs_embeds = base_embeddings
        
        # Generate with the transformer
        generated_ids = self.transformer.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        return generated_ids
    
    def get_config(self) -> Dict:
        return self.config.copy()
    
    def count_parameters(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_dynamo_model(transformer_path: str, **kwargs) -> DynamoModel:
    default_config = {
        'transformer': transformer_path,
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
    
    # Test the model with sample inputs
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    time = torch.randn(batch_size)
    edge_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, time, edge_indices, labels=labels)
        print(f"Loss: {outputs.loss}")
        print(f"Logits shape: {outputs.logits.shape}")
