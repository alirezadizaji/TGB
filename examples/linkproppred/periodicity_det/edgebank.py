"""
Dynamic Link Prediction with EdgeBank
NOTE: This implementation works only based on `numpy`

Reference: 
    - https://github.com/fpour/DGB/tree/main


"""

import timeit
import numpy as np
from tqdm import tqdm
import math
import os
import os.path as osp
from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import networkx as nx
import torch

# internal imports
from ....tgb.linkproppred.evaluate import Evaluator
from ....modules.edgebank_predictor import EdgeBankPredictor
from ....tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from ....tgb.utils.utils import set_random_seed
from ....tgb.utils.utils import save_results

# ==================
def visualizer(save_dir: str):
    def _visualize(pred_adj: torch.Tensor, target_adj: torch.Tensor, filename: str):
        os.makedirs(save_dir, exist_ok=True)
        pred_adj = (pred_adj >= 0.5)
        pred_src, pred_dst = np.nonzero(pred_adj)

        target_src, target_dst = np.nonzero(target_adj)

        G1 = nx.Graph()
        G1.add_nodes_from(list(range(50)))
        G1.add_edges_from(list(zip(target_src, target_dst)))
        pos = nx.kamada_kawai_layout(G1,)
        _, axes = plt.subplots(1, 2, figsize=(20, 10))
        nx.draw_networkx(G1, pos, node_size=200, node_color='lightblue', ax=axes[0])
        axes[0].set_title(f"Target", fontsize=30)
        G2 = nx.Graph()
        G2.add_nodes_from(list(range(50)))
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


def test(data, test_mask, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        data: a dataset object
        test_mask: required masks to load the test set edges
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    vis = visualizer(save_dir=os.path.join(results_path, MODEL_NAME, DATA, split_mode, MEMORY_MODE, str(TIME_WINDOW_RATIO)))

    perf_list = []
    pred_adj = np.zeros_like(adj)
    
    if split_mode == "val":
        start_idx, end_idx = val_start_t, test_start_t
    elif split_mode == "test":
        start_idx, end_idx = test_start_t, max_t + 1
    
    for ct in tqdm(range(start_idx, end_idx)):
        mask = (t == ct)
        pos_t = t[mask]
        pos_src = src[mask]
        pos_dst = dst[mask]

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])

            y_pred = edgebank.predict_link(query_src, query_dst)
            pred_adj[pos_t[idx], pos_src[idx], pos_dst[idx]] = y_pred[0]
            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0]]),
                "y_pred_neg": np.array(y_pred[1:]),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # update edgebank memory after each positive batch
        edgebank.update_memory(pos_src, pos_dst, pos_t)


    for i, idx in enumerate(range(start_idx, end_idx)):
        vis(pred_adj[idx], adj[idx], str(i))

    perf_metrics = float(np.mean(perf_list))

    return perf_metrics

def get_args():
    parser = argparse.ArgumentParser('*** TGB: EdgeBank ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-comment')
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_mode', type=str, help='Memory mode', default='unlimited', choices=['unlimited', 'fixed_time_window'])
    parser.add_argument('--time_window_ratio', type=float, help='Test window ratio', default=0.15)
    parser.add_argument('-l', '--data-loc', type=str, help='The location where data is stored.')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

# ==================
# ==================
# ==================

start_overall = timeit.default_timer()

# set hyperparameters
args, _ = get_args()

SEED = args.seed  # set the random seed for consistency
set_random_seed(SEED)
MEMORY_MODE = args.mem_mode # `unlimited` or `fixed_time_window`
BATCH_SIZE = args.bs
K_VALUE = args.k_value
TIME_WINDOW_RATIO = args.time_window_ratio
DATA = args.data
DATA_LOC = args.data_loc

MODEL_NAME = 'EdgeBank'

# data loading with `numpy`
assert os.path.exists(args.data_loc), f"The given data location does not exist: {args.data_loc}"
data = np.load(os.path.join(DATA_LOC, DATA, "data.npz"))
metric = "mrr"

# get masks
src = data["src"]
dst = data["dst"]
t = data["t"]
train_mask = data["train_mask"]
val_mask = data["val_mask"]
test_mask = data["test_mask"]

max_t = t.max()

if "num_nodes" in data:
    num_nodes = data["num_nodes"]
else:
    num_nodes = data["node_feat"].shape[0]

adj = np.zeros((max_t + 1, num_nodes, num_nodes))
adj[t, src, dst] = 1

val_start_t = t[val_mask].min()
test_start_t = t[test_mask].min()

# Separating train/val/test edge indices. "1/-1" is added to let each set also predicts the output of last day of week
train_adj = adj[:val_start_t]
val_adj = adj[val_start_t: test_start_t]
test_adj = adj[test_start_t:]

#data for memory in edgebank
hist_src = np.concatenate([src[train_mask]])
hist_dst = np.concatenate([dst[train_mask]])
hist_ts = np.concatenate([t[train_mask]])

# Set EdgeBank with memory updater
edgebank = EdgeBankPredictor(
        hist_src,
        hist_dst,
        hist_ts,
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO)

print("==========================================================")
print(f"============*** {MODEL_NAME}: {MEMORY_MODE}: {TIME_WINDOW_RATIO}: {DATA} ***==============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = NegativeEdgeSampler(dataset_name=DATA)

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{MEMORY_MODE}_{DATA}_results.json'

# ==================================================== Test
# loading the validation negative samples
neg_sampler.load_eval_set(os.path.join(DATA_LOC, DATA, "val_ns.pkl"), split_mode="val")

# testing ...
start_val = timeit.default_timer()
perf_metric_test = test(data, val_mask, neg_sampler, split_mode='val')
end_val = timeit.default_timer()

print(f"INFO: val: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tval: {metric}: {perf_metric_test: .4f}")
test_time = timeit.default_timer() - start_val
print(f"\tval: Elapsed Time (s): {test_time: .4f}")




# ==================================================== Test
# loading the test negative samples
neg_sampler.load_eval_set(os.path.join(DATA_LOC, DATA, "test_ns.pkl"), split_mode="test")

# testing ...
start_test = timeit.default_timer()
perf_metric_test = test(data, test_mask, neg_sampler, split_mode='test')
end_test = timeit.default_timer()

print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tTest: {metric}: {perf_metric_test: .4f}")
test_time = timeit.default_timer() - start_test
print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

save_results({'model': MODEL_NAME,
              'memory_mode': MEMORY_MODE,
              'data': DATA,
              'run': 1,
              'seed': SEED,
              metric: perf_metric_test,
              'test_time': test_time,
              'tot_train_val_time': 'NA',
              'time_window_ratio': TIME_WINDOW_RATIO,
              }, 
    results_filename)
