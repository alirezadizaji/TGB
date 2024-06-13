from typing_extensions import Protocol

import torch

class RecurrentGNN(Protocol):
    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ...