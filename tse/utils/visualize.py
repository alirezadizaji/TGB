import os
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

def visualizer(save_dir: str, num_nodes: int) -> Callable[[torch.Tensor, torch.Tensor, str], None]:
    def _visualize(pred_adj: torch.Tensor, target_adj: torch.Tensor, filename: str):
        os.makedirs(save_dir, exist_ok=True)
        pred_adj = (pred_adj >= 0.5).detach().cpu().numpy()
        pred_src, pred_dst = np.nonzero(pred_adj)

        target_adj = target_adj.detach().cpu().numpy()
        target_src, target_dst = np.nonzero(target_adj)

        G1 = nx.Graph()
        G1.add_nodes_from(list(range(num_nodes)))
        G1.add_edges_from(list(zip(target_src, target_dst)))
        pos = nx.kamada_kawai_layout(G1,)
        _, axes = plt.subplots(1, 2, figsize=(20, 10))
        nx.draw_networkx(G1, pos, node_size=200, node_color='lightblue', ax=axes[0])
        axes[0].set_title(f"Target", fontsize=30)
        G2 = nx.Graph()
        G2.add_nodes_from(list(range(num_nodes)))
        G2.add_edges_from(list(zip(pred_src, pred_dst)))
        nx.draw_networkx(G2, pos, node_size=200, node_color='lightblue', ax=axes[1])
        axes[1].set_title(f"Prediction", fontsize=30)
        
        d = os.path.join(save_dir, "vis")
        os.makedirs(d, exist_ok=True)
        d = os.path.join(d, filename)
        plt.suptitle(filename, fontsize=40)
        plt.savefig(d)
        plt.close()

    return _visualize