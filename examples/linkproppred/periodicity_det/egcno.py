"""
Dynamic Link Prediction with a EvolveGCNO model with Early Stopping
"""

import argparse
import timeit
import os
import os.path as osp
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from ....tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from ....tse.dataset.dataset import SnapshotDataset
from ....tgb.utils.utils import set_random_seed, save_results
from ....tgb.linkproppred.evaluate import Evaluator
from ....tse.models import MultiLayerEGCNO, EvolveGCNParams
from ....modules.decoder import LinkPredictor
from ....modules.early_stopping import  EarlyStopMonitor


# ==========
# ========== Define helper function...
# ==========

def get_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    # parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=50)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--num_units', type=int, help='Number of EvolveGCN units', default=1)
    parser.add_argument('--in_channels', type=int, help='input channel dimension of EvolveGCNO')
    parser.add_argument('--improved', type=bool, help='If True, then identity is added to adjacency matrix', default=False)
    parser.add_argument('--cached', type=bool, help='If True, then EvolveGCN caches the normalized adjacency matrix and uses it in next steps', default=False)
    parser.add_argument('--normalize', type=bool, help='If True, then EvolveGCN normalizes the adjacency matrix', default=True)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=5)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)
    parser.add_argument('-l', '--data-loc', type=str, help='The location where data is stored.')

    try:
        args = parser.parse_known_args()[0]
    except:
        parser.print_help()
        sys.exit(0)
        
    return args, sys.argv


def train():
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['gnn'].train()
    model['link_pred'].train()
    total_loss = 0
    for i in range(len(train_set)):
        
        start_edge_index, end_edge_index, curr_t = train_set[i]
        start_edge_index = start_edge_index.to(device)
        end_edge_index = end_edge_index.to(device)

        optimizer.zero_grad()

        z = model['gnn'](
            start_edge_index,
            node_feat)
        
        out: torch.Tensor = model['link_pred'](z)             # N, 1
        out_2d = out * out.T                                  # N, N

        loss = criterion(out_2d, end_edge_index)

        loss.backward()
        optimizer.step()
        total_loss += float(loss)

    return total_loss / len(train_set), out_2d


@torch.no_grad()
def test(dataset: SnapshotDataset, neg_sampler: torch.Tensor, split_mode: torch.Tensor, out_2d: torch.Tensor):
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    out_2d = out_2d.detach()
    
    for i in range(len(dataset)):
        adjacency = (out_2d >= 0.5)
        s, d = torch.nonzero(adjacency)
        edge_index = torch.stack([s, d], dim=1)
        _, _, curr_t = train_set[i]
        # start_edge_index, end_edge_index, curr_t = train_data[i]
        # start_edge_index = start_edge_index.to(device)
        # end_edge_index = end_edge_index.to(device)

        mask = (t == curr_t)
        pos_src = src[mask]
        pos_dst = dst[mask]
        pos_t = t[mask]

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)


        z = model['gnn'](
            edge_index,
            node_feat)
        out = model['link_pred'](z)
        out_2d = out * out.T

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            ).long()


            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([out_2d[src, dst[0]].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(out_2d[src, dst[1:]].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()

print("INFO: Arguments:", args)

DATA = 'tgbl-comment'
LR = args.lr
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
EMB_DIM = args.in_channels
NUM_UNITS = args.num_units
SEED = args.seed
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run

# Custom variables
DATA_LOC = args.data_loc


MODEL_NAME = 'EGCNO'
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################ DATA LOADING ###################
assert os.path.exists(args.data_loc), f"The given data location does not exist: {args.data_loc}"
data_np = np.load(os.path.join(DATA_LOC, DATA, "data.npz"))
src = torch.tensor(data_np["src"])
dst = torch.tensor(data_np["dst"])
t = torch.tensor(data_np["t"])
edge_feat = torch.tensor(data_np["edge_feat"])
node_feat = torch.tensor(data_np["node_feat"])
train_mask = list(data_np["train_mask"])
val_mask = list(data_np["val_mask"])
test_mask = list(data_np["test_mask"])

max_t = t.max()

num_nodes = node_feat.size(0)
adj = torch.zero((max_t, num_nodes, num_nodes))
adj[t, src, dst] = 1

val_start_t = t[val_mask].min()
test_start_t = t[test_mask].min()

# Separating train/val/test edge indices. "1/-1" is added to let each set also predicts the output of last day of week
train_adj = adj[:val_start_t]
val_adj = adj[val_start_t: test_start_t + 1]
test_adj = adj[test_start_t-1:]

train_set = SnapshotDataset(train_adj)
val_set = SnapshotDataset(val_adj)
test_set = SnapshotDataset(test_adj)

metric = "mrr"

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(dst.min()), int(dst.max())

##############  MODELS  ############
gnn = MultiLayerEGCNO(
    num_units=NUM_UNITS, 
    base_args=EvolveGCNParams(
        EMB_DIM,
        args.improved,
        args.cached,
        args.normalize))
link_pred = torch.nn.Sequential(
    torch.nn.Linear(EMB_DIM, EMB_DIM),
    torch.nn.ReLU(),
    torch.nn.Linear(EMB_DIM, 1),
    torch.nn.Sigmoid())

model = {'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCELoss()

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = NegativeEdgeSampler(dataset_name=DATA)

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples
    neg_sampler.load_eval_set(os.path.join(DATA_LOC, DATA, "val_ns.pkl"), split_mode="val")

    val_perf_list = []
    train_times_l, val_times_l = [], []
    free_mem_l, total_mem_l, used_mem_l = [], [], []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss = train()
        end_epoch_train = timeit.default_timer()
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}"
        )
        # checking GPU memory usage
        free_mem, used_mem, total_mem = 0, 0, 0
        if torch.cuda.is_available():
            print("DEBUG: device: {}".format(torch.cuda.get_device_name(0)))
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            print("------------Epoch {}: GPU memory usage-----------".format(epoch))
            print("Free memory: {}".format(free_mem))
            print("Total available memory: {}".format(total_mem))
            print("Used memory: {}".format(used_mem))
            print("--------------------------------------------")
        
        train_times_l.append(end_epoch_train - start_epoch_train)
        free_mem_l.append(float((free_mem*1.0)/2**30))  # in GB
        used_mem_l.append(float((used_mem*1.0)/2**30))  # in GB
        total_mem_l.append(float((total_mem*1.0)/2**30))  # in GB

        # validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_set, neg_sampler, split_mode="val")
        end_val = timeit.default_timer()
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {end_val - start_val: .4f}")
        val_perf_list.append(perf_metric_val)
        val_times_l.append(end_val - start_val)

        # check for early stopping
        if early_stopper.step_check(perf_metric_val, model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

    # loading the test negative samples
    neg_sampler.load_eval_set(os.path.join(DATA_LOC, DATA, "test_ns.pkl"), split_mode="test")


    # final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(test_set, neg_sampler, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results({'data': DATA,
                  'model': MODEL_NAME,
                  'run': run_idx,
                  'seed': SEED,
                  'train_times': train_times_l,
                  'free_mem': free_mem_l,
                  'total_mem': total_mem_l,
                  'used_mem': used_mem_l,
                  'max_used_mem': max(used_mem_l),
                  'val_times': val_times_l,
                  f'val {metric}': val_perf_list,
                  f'test {metric}': perf_metric_test,
                  'test_time': test_time,
                  'train_val_total_time': np.sum(np.array(train_times_l)) + np.sum(np.array(val_times_l)),
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
