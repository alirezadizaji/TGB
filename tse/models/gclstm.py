from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import GCLSTM

@dataclass
class GCLSTMParam:
    in_channels: int
    out_channels: int
    K: int = 1

class MultiLayerGCLSTM(nn.Module):
    def __init__(self, num_units: int, gclstm_param: GCLSTMParam) -> None:
        super(MultiLayerGCLSTM, self).__init__()
        self.num_units = num_units

        self.units = nn.ModuleList([])
        self.units.append(GCLSTM(
            **asdict(gclstm_param)
        ))

        for _ in range(1, self.num_units):
            self.units.append(GCLSTM(
                in_channels=gclstm_param.out_channels,
                out_channels=gclstm_param.out_channels,
                K=gclstm_param.K,
            ))
    
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, hs: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
        inp = X
        out_hs = []
        out_cs = []

        for i in range(self.num_units):
            h, c = self.units[i](inp, edge_index, edge_weight=None, H=hs[i], C=cs[i])
            out_hs.append(h)
            out_cs.append(c)
            inp = h
        
        h1 = torch.stack(out_hs)
        c1 = torch.stack(out_cs)
        return h1, c1, h