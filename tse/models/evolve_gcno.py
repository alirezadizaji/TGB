from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch_geometric_temporal import EvolveGCNO

from .recurrent_gnn import RecurrentGNN

@dataclass
class EvolveGCNParams:
    in_channels: int
    improved: bool = False
    cached: bool = False
    normalize: bool = True

class MultiLayerEGCNO(RecurrentGNN):
    def __init__(self, num_units: int, base_args: EvolveGCNParams) -> None:
        super(MultiLayerEGCNO, self).__init__()

        self.base_args = base_args
        self.units: nn.ModuleList[EvolveGCNO] = nn.ModuleList([])
        for _ in range(num_units):
            self.units.append( 
                EvolveGCNO(**asdict(self.base_args), add_self_loops=False)
                )
        
    def forward(self, edge_index: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            X = unit(edge_index, X)
        
        return X