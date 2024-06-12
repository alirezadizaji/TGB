""" This file generates datasets which models weekly periodicity structure in temporal graphs.
"""

import argparse
from importlib import import_module
import os
import random
import pickle
import sys
from types import SimpleNamespace
import torch
import yaml

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.data import TemporalData

from ...tgb.linkproppred.negative_generator import NegativeEdgeGenerator

from .negative_sampler import gen_neg_dst

def _get_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('-c', '--conf-dir', type=str, help='Configuration directory for graph generation')
    parser.add_argument('-s', '--save-dir', type=str, help='Save directory', default=1e-4)

    args = parser.parse_args()
    return args

def to_adj(A: np.ndarray, num_nodes: int) -> np.ndarray:
    Adj = np.zeros((num_nodes, num_nodes))
    Adj[A[:, 0], A[:, 1]] = 1
    return Adj

def to_sparse(Adj: np.ndarray) -> np.ndarray:
    r, c = np.nonzero(Adj)
    A = np.stack([r, c], axis=1)
    return A

def main():
    """ This function generates synthetic temporal data which models (weekly) periodicity, using determinstic patterns. 
    The negative edge sampling method is adopted from TGB framework by Huang et al (2023).
    """
    args = _get_args()
    
    with open(args.conf_dir, "r") as f:
        config = yaml.safe_load(f)
        config = SimpleNamespace(**config)

    for data in config.data:
        try:
            name = data["name"]
            num_nodes = data["num_nodes"]
            train_num_weeks = data["train_num_weeks"]
            seed = data["seed"]
            verbose = data["verbose"]
            neg_sampling = data["neg_sampling"]
            graphs = data["graphs"]
            directed = data["directed"]
            permute = data["permute_nodes"]
        
            fdir = os.path.join(args.save_dir, name)
            if os.path.exists(fdir):
                print(f"\n\n=====> Skipping Data generation for {name}. Delete this directory {fdir} for regeneration", flush=True)
                continue
        
        except Exception as e:
            raise ValueError(f"Invalid configuration for {args.conf_dir}. {e}")
        
        random.seed(seed)
        np.random.seed(seed)

        print(f"Generating dataset {name}...", flush=True)
        
        day_sample = dict()
        for g in graphs:

            A: np.ndarray = None
            for d in g["days"]:
                if d in day_sample:
                    raise Exception(f"Invalid configuration. Day {d} has more than one deterministic graph patterns.")
                
                if A is not None:
                    day_sample[d] = A
                    continue
                
                func = getattr(nx, g["pattern"])
                sample: nx.Graph = func(**g["params"])
            
                A = np.array(sample.edges)

                # Make sure undirected graphs have bilateral edges
                if directed == False:
                    A = np.concatenate([A, A[:, [1, 0]]], axis=0)
                    A = np.unique(A, axis=0)

                # Permute nodes randomly
                if permute:
                    I = np.eye(num_nodes)
                    P = I
                    Adj = to_adj(A, num_nodes=num_nodes)
                    while np.allclose(P, I):
                        ind = np.random.choice(np.arange(num_nodes), size=num_nodes, replace=False)
                        P = I[ind]
                    
                    # permutation
                    Adj = P @ Adj @ P.T

                    A = to_sparse(Adj)
                
                day_sample[d] = A
        
        src = np.empty(0)
        dst = np.empty_like(src)
        t = np.empty_like(src)
        edge_feat = np.empty_like(src)

        # Generate for only one week
        for day in range(7):
            A: np.ndarray = day_sample[day]
            if verbose:
                print("\tVisualizing...")
                
                # Create the graph
                G = nx.Graph()
                G.add_nodes_from(np.arange(num_nodes))
                G.add_edges_from(A)

                pos = nx.kamada_kawai_layout(G)
                nx.draw_networkx(G, pos, node_size=30, with_labels=True, node_color="yellow")
                plt.title(f"Name {name} Day {day}")
                plt.show(block=False)
                plt.pause(3)
                plt.close()

            num_edges = A.shape[0]

            src = np.concatenate([src, A[:, 0]])
            dst = np.concatenate([dst, A[:, 1]])
            t = np.concatenate([t, np.repeat(day, num_edges)])
            edge_feat = np.concatenate([edge_feat, np.repeat(1, num_edges)])

        one_week_num_samples = t.size

        # repeat generated samples in multiple weeks + 2 additional weeks (1 for val, 1 for test)
        all_weeks =  train_num_weeks + 2
        src = np.tile(src, all_weeks).astype(np.int64)
        dst = np.tile(dst, all_weeks).astype(np.int64)
        edge_feat = np.tile(edge_feat, all_weeks)[:, np.newaxis].astype(np.float32)
        ## DO NOT REORDER FOLLOWING LINES
        start = np.arange(all_weeks) * 7
        start = np.repeat(start, t.size)
        t = np.tile(t, all_weeks).astype(np.int64)
        t = t + start

        sample_id = np.arange(t.size)
        test_mask = np.roll(sample_id < one_week_num_samples, -one_week_num_samples)
        val_mask = np.roll(test_mask, -one_week_num_samples)
        train_mask = (1 - val_mask - test_mask) == 1

        print(f"\t Data generated. src shape: {src.shape}, edge_feat shape: {edge_feat.shape}", flush=True)
        data = TemporalData(
            src=torch.tensor(src),
            dst=torch.tensor(dst),
            t=torch.tensor(t),
            msg=torch.tensor(edge_feat))

        data_splits = {}
        data_splits['train'] = data[torch.tensor(train_mask)]
        data_splits['val'] = data[torch.tensor(val_mask)]
        data_splits['test'] = data[torch.tensor(test_mask)]

        # Ensure to only sample actual destination nodes as negatives.
        min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

        # After successfully loading the dataset...
        if neg_sampling["strategy"] == "hist_rnd":
            historical_data = data_splits["train"]
        else:
            historical_data = None
        neg_generator = NegativeEdgeGenerator(
            dataset_name=name,
            first_dst_id=min_dst_idx,
            last_dst_id=max_dst_idx,
            num_neg_e=neg_sampling["num_neg_edge"],
            strategy=neg_sampling["strategy"],
            rnd_seed=seed,
            historical_data=historical_data,
        )


        # generate validation negative edge set
        os.makedirs(fdir, exist_ok=True)
        import time
        start_time = time.time()
        split_mode = "val"
        print(
            f"INFO: Start generating negative samples: {split_mode} --- {neg_sampling['strategy']}"
        )
        neg_generator.generate_negative_samples(
            data=data_splits[split_mode], split_mode=split_mode, partial_path=fdir
        )
        print(
            f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
        )

        # generate test negative edge set
        start_time = time.time()
        split_mode = "test"
        print(
            f"INFO: Start generating negative samples: {split_mode} --- {neg_sampling['strategy']}"
        )
        neg_generator.generate_negative_samples(
            data=data_splits[split_mode], split_mode=split_mode, partial_path=fdir
        )
        print(
            f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
        )

        node_feat = np.ones((num_nodes, 1)).astype(np.float32)
        np.savez_compressed(os.path.join(fdir, "data.npz"), 
            src=src, 
            dst=dst, 
            t=t,
            edge_feat=edge_feat,
            node_feat=node_feat,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask)


if __name__ == "__main__":
    main()