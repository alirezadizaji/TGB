""" This file generates datasets which models weekly periodicity structure in temporal graphs.
"""

import os
import random
import pickle
import time
from tqdm import tqdm
from types import SimpleNamespace
import yaml

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

from tse.dataset.negative_sampler import gen_neg_dst

def main():
    yaml_fname = "ER-weekly-periodicity.yml"
    with open(f"./config/{yaml_fname}", "r") as f:
        config = yaml.safe_load(f)
        config = SimpleNamespace(**config)

    for data in config.data:
        try:
            name = data["name"]
            num_nodes = data["num_nodes"]
            duration = data["duration"]
            prob = data["p"]
            seed = data["seed"]
            verbose = data["verbose"]
            directed = data["directed"]
            val_split = data["val_split"]
            test_split = data["test_split"]
            num_neg_edges = data["num_neg_edge"]
        except Exception as e:
            raise ValueError(f"Invalid configuration for {yaml_fname}. {e}")
        
        random.seed(seed)
        np.random.seed(seed)

        print(f"Generating dataset {name}...", flush=True)
        
        src = np.empty(0)
        dst = np.empty_like(src)
        t = np.empty_like(src)
        edge_feat = np.empty_like(src)

        for i in tqdm(range(duration)):
            day = i % 7
            p = prob[day]

            # Generating directed Erdos-Renyi graph
            G = nx.erdos_renyi_graph(num_nodes, p=p, directed=directed)
            if verbose and i < 7:
                pos = nx.kamada_kawai_layout(G)
                nx.draw_networkx_edges(G, pos)
                nx.draw_networkx_nodes(G, pos, node_size=10)
                plt.title(f"Name {name} Day {day}")
                plt.show(block=False)
                plt.pause(1)
                plt.close()

            A = np.array(G.edges)
            num_edges = A.shape[0]

            src = np.concatenate([src, A[:, 0]])
            dst = np.concatenate([dst, A[:, 1]])
            t = np.concatenate([t, np.repeat(i, num_edges)])
            edge_feat = np.concatenate([edge_feat, np.repeat(1, num_edges)])

        # TODO: Node feature generation confirmation
        node_feat = np.ones((num_nodes, 1))
        
        num_samples = t.size
        sample_id = np.arange(num_samples)
        val_start_idx = int((1. - val_split - test_split) * num_samples)
        test_start_idx = int((1 - test_split) * num_samples)

        train_mask = (sample_id < val_start_idx)
        val_mask = np.logical_and(sample_id >= val_start_idx, sample_id < test_start_idx)
        test_mask = (sample_id >= test_start_idx)

        val_ns = gen_neg_dst(src[val_mask], dst[val_mask], t[val_mask], num_nodes, num_neg_edges)
        test_ns = gen_neg_dst(src[test_mask], dst[test_mask], t[test_mask], num_nodes, num_neg_edges)

        save_dir = os.path.join(config.save_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(os.path.join(save_dir, "data.npz"), 
            src=src, 
            dst=dst, 
            t=t,
            edge_feat=edge_feat,
            node_feat=node_feat,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask)

        with open(os.path.join(save_dir, 'val_ns.pkl'), 'wb') as handle:
            pickle.dump(val_ns, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
        with open(os.path.join(save_dir, 'test_ns.pkl'), 'wb') as handle2:
            pickle.dump(test_ns, handle2, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()


