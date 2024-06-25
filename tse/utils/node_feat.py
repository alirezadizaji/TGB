from typing import Any

import torch

from ..types import NodeFeatType

class NodeFeatGenerator:
    def __init__(self, type: NodeFeatType, emb_dim: int):
        self.type = type
        self.emb_dim: int = emb_dim
    
    def __call__(self, num_nodes: int) -> torch.Tensor:
        node_feat = None

        if self.type == NodeFeatType.CONSTANT:
            node_feat = torch.ones((num_nodes, 1), dtype=torch.float32)
        elif self.type == NodeFeatType.RAND:
            node_feat = torch.rand((num_nodes, self.emb_dim), dtype=torch.float32)
        elif self.type == NodeFeatType.ONE_HOT:
            node_feat = torch.eye(num_nodes)
        elif self.type == NodeFeatType.NODE_ID:
            node_feat = torch.arange(num_nodes).unsqueeze(1).float()
        
        print(f"===========> Node feature generated: size {node_feat.size()}", flush=True)
        return node_feat