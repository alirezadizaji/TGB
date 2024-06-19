from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch_geometric_temporal import EvolveGCNO

@dataclass
class EvolveGCNParams:
    in_channels: int
    improved: bool = False
    cached: bool = False
    normalize: bool = True

class MultiLayerEGCNO(torch.nn.Module):
    def __init__(self, num_units: int, base_args: EvolveGCNParams, inp_dim: int) -> None:
        super(MultiLayerEGCNO, self).__init__()

        self.base_args = base_args
        self.units: nn.ModuleList[EvolveGCNO] = nn.ModuleList([])
        for _ in range(num_units):
            self.units.append( 
                EvolveGCNO(**asdict(self.base_args), add_self_loops=False)
                )
        
        self.lin = nn.Linear(inp_dim, base_args.in_channels)
        
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        X = self.lin(X)
        
        for unit in self.units:
            X = unit(X, edge_index)
        
        return X