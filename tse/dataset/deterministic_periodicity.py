""" This file generates datasets which models weekly periodicity structure in temporal graphs.
"""

from importlib import import_module
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
    yaml_fname = "deterministic-periodicity.yml"
    with open(f"./config/{yaml_fname}", "r") as f:
        config = yaml.safe_load(f)
        config = SimpleNamespace(**config)

    for data in config.data:
        try:
            name = data["name"]
            num_nodes = data["num_nodes"]
            train_num_weeks = data["train_num_weeks"]
            seed = data["seed"]
            verbose = data["verbose"]
            directed = data["directed"]
            num_neg_edges = data["num_neg_edge"]
            graphs = data["graphs"]
        except Exception as e:
            raise ValueError(f"Invalid configuration for {yaml_fname}. {e}")
        
        random.seed(seed)
        np.random.seed(seed)

        print(f"Generating dataset {name}...", flush=True)
        
        day_sample = dict()
        for g in graphs:

            sample = None
            for d in g["days"]:
                if d in day_sample:
                    raise Exception(f"Invalid configuration. Day {d} has more than one deterministic graph patterns.")
                
                if sample is None:
                    func = getattr(nx, g["pattern"])
                    sample = func(**g["params"])
                
                day_sample[d] = sample
        
        src = np.empty(0)
        dst = np.empty_like(src)
        t = np.empty_like(src)
        edge_feat = np.empty_like(src)

        # Generate for only one week, then repeat the pattern
        for day in range(7):
            G = day_sample[day]

            if verbose:
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
            t = np.concatenate([t, np.repeat(day, num_edges)])
            edge_feat = np.concatenate([edge_feat, np.repeat(1, num_edges)])

        one_week_num_samples = t.size

        # repeat generated samples in multiple weeks + 2 additional weeks (1 for val, 1 for test)
        rest_num_weeks =  ((train_num_weeks + 2) - 1)
        src = np.tile(src, rest_num_weeks)
        dst = np.tile(dst, rest_num_weeks)
        edge_feat = np.tile(edge_feat, rest_num_weeks)
        ## DO NOT REORDER FOLLOWING LINES
        start = np.arange(rest_num_weeks + 1) * 7
        start = np.repeat(start, t.size)
        t = t + start

        sample_id = np.arange(t.size)
        test_mask = np.roll(sample_id <= one_week_num_samples, -one_week_num_samples)
        val_mask = np.roll(test_mask, -one_week_num_samples)
        train_mask = (1 - val_mask - test_mask) == 1
        val_ns = gen_neg_dst(src[val_mask], dst[val_mask], t[val_mask], num_nodes, num_neg_edges)
        test_ns = gen_neg_dst(src[test_mask], dst[test_mask], t[test_mask], num_nodes, num_neg_edges)

        node_feat = np.ones((num_nodes, 1))

        save_dir = os.path.join(config.save_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(os.path.join(save_dir, "data.npz"), 
            src=src, 
            dst=dst, 
            t=t,
            edge_feat=edge_feat,
            node_feat=node_feat,
            val_mask=val_mask,
            test_mask=test_mask)

        with open(os.path.join(save_dir, 'val_ns.pkl'), 'wb') as handle:
            pickle.dump(val_ns, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
        with open(os.path.join(save_dir, 'test_ns.pkl'), 'wb') as handle2:
            pickle.dump(test_ns, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


