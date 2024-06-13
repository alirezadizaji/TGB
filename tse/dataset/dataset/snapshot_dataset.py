from typing import List

from torch.utils.data import Dataset
import torch

class SnapshotDataset(Dataset):
    def __init__(self, adj: torch.Tensor, start_time: int) -> None:
        """
        Args:
            adj (torch.Tensor): adjacency matrix representing graph structure in different time steps. (Shape: T, N, N) 
            start_time (List[int]): Starting time that represents the first adjacency matrix. 
        """
        super().__init__()

        self.adj = adj
        self.times = start_time + torch.arange(self.adj.size(0))
    
    def __len__(self):
        return self.adj.size(0)
    
    def __getitem__(self, index) -> torch.Tensor:
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, int):
            index = [index]

        assert len(index) == 1, "SnapshotDataset only supports batch-size of 1 due to different sizes of edge indices between samples." 

        i = index[0]
        A = self.adj[i]
        isrc, idst = torch.nonzero(A) 
        inp_edge_index = torch.stack([isrc, idst], dim=1)      

        A2 = self.adj[i + 1]
        osrc, odst = torch.nonzero(A2)
        out_edge_index = torch.stack([osrc, odst], dim=1)

        return inp_edge_index, out_edge_index, self.times[i]