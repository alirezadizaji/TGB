"""
Dynamic Link Prediction with a EvolveGCNO model with Early Stopping
"""

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import timeit
import os
import os.path as osp
from pathlib import Path
import sys

import numpy as np
import torch

from ....tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from ....tgb.utils.utils import set_random_seed, save_results
from ....tgb.linkproppred.evaluate import Evaluator
from ....tse import MultiLayerEGCNO, EvolveGCNParams, LinkPredictor, NodeFeatType, NodeFeatGenerator
from ....modules.early_stopping import  EarlyStopMonitor


# ==========
# ========== Define helper function...
# ==========

def get_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--k-value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num-epoch', type=int, help='Number of epochs', default=50)
    parser.add_argument('--node-feat', choices=NodeFeatType.list(), help='Type of node feature generation', default=NodeFeatType.CONSTANT)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--num-units', type=int, help='Number of EvolveGCN units', default=1)
    parser.add_argument('--in-channels', type=int, help='input channel dimension of EvolveGCNO', default=100)
    parser.add_argument('--improved', type=bool, help='If True, then identity is added to adjacency matrix', default=False)
    parser.add_argument('--cached', type=bool, help='If True, then EvolveGCN caches the normalized adjacency matrix and uses it in next steps', default=False)
    parser.add_argument('--normalize', type=bool, help='If True, then EvolveGCN normalizes the adjacency matrix', default=True)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=5)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--num-run', type=int, help='Number of iteration runs', default=1)
    parser.add_argument('-l', '--data-loc', type=str, help='The location where data is stored.')

    try:
        args = parser.parse_known_args()[0]
    except:
        parser.print_help()
        sys.exit(0)
        
    return args, sys.argv


def visualizer(save_dir: str):
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

def train():
    r"""
    Training procedure for EvolveGCN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """
    # This variable stores the model output. All elements should have a valid number at the end
    out_2d = torch.zeros((num_nodes, num_nodes), requires_grad=False)

    model['gnn'].train()
    model['link_pred'].train()
    total_loss = 0
    for cur_t, trainA in enumerate(train_adj):
   
        prev_edge_index = None
        # At time step 0, the input graph is empty of edges
        if cur_t == 0:
            prev_src, prev_dst = torch.nonzero(torch.zeros_like(trainA), as_tuple=True)
        else:
            prev_src, prev_dst = torch.nonzero(train_adj[cur_t - 1], as_tuple=True)
        
        prev_edge_index = torch.stack([prev_src, prev_dst], dim=0)
        
        # Separate positive and negative pairs
        cur_src, cur_dst = torch.nonzero(trainA, as_tuple=True)

        # to prevent considering self-loop edges as negative pairs, temporarily set the diagonal value of trainA as nonzero
        trainA.fill_diagonal_(1)
        neg_src, neg_dst = torch.nonzero(trainA == 0, as_tuple=True)
        trainA.fill_diagonal_(0)

        prev_edge_index = prev_edge_index.to(device)

        optimizer.zero_grad()

        z = model['gnn'](
            node_feat,
            prev_edge_index.long())

        pos_pred: torch.Tensor = model['link_pred'](z[cur_src], z[cur_dst])
        neg_pred: torch.Tensor = model['link_pred'](z[neg_src], z[neg_dst])

        # Sigmoid assertion
        assert torch.all(pos_pred <= 1) and torch.all(pos_pred >= 0), "Sigmoid assertion failed. make sure `sigmoid` is applied at model output."

        # Loss calculation
        loss = criterion(pos_pred, torch.ones_like(pos_pred))
        loss += criterion(neg_pred, torch.zeros_like(neg_pred))

        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())

        # Predicted edge index
        out_2d.fill_(torch.nan)
        out_2d[cur_src, cur_dst] = pos_pred.squeeze(-1).detach()
        out_2d[neg_src, neg_dst] = neg_pred.squeeze(-1).detach()
        out_2d.fill_diagonal_(0)  # Model does not predict self-loop edges

        # Valid number assertion
        assert torch.all(out_2d != torch.nan)

    return total_loss / train_adj.size(0), out_2d


@torch.no_grad()
def test(neg_sampler: torch.Tensor, split_mode: torch.Tensor, out_2d: torch.Tensor, start_time: int):
    model['gnn'].eval()
    model['link_pred'].eval()

    vis = visualizer(save_dir=os.path.join(results_path, MODEL_NAME, DATA, split_mode, f"NODEFEAT-{args.node_feat}_UNIT-{NUM_UNITS}_EMB-{EMB_DIM}", str(epoch)))

    perf_list = []

    out_2d = out_2d.detach()
    
    for i, evalA in enumerate(val_adj):
        cur_t = i + start_time

        prev_edge_index = None
        # At first step of evaluation, pass the last time recurrent model output as the first adjacency input
        if i == 0:
            prev_adj = (out_2d >= 0.5).long()
        else:
            prev_adj = val_adj[i]
        prev_src, prev_dst = torch.nonzero(prev_adj, as_tuple=True)
        prev_edge_index = torch.stack([prev_src, prev_dst], dim=0)
        
        # Separate positive and negative pairs
        pos_src, pos_dst = torch.nonzero(evalA, as_tuple=True)

        # To prevent considering self-loop edges as negative pairs, set diagonal elements as zero.
        evalA.fill_diagonal_(1)
        neg_src, neg_dst = torch.nonzero(evalA == 0, as_tuple=True)
        evalA.fill_diagonal_(0)

        pos_t = t[t == cur_t]
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        prev_edge_index = prev_edge_index.to(device)
        z = model['gnn'](
            node_feat,
            prev_edge_index)

        # Predicted edge index
        pos_pred = model['link_pred'](z[pos_src], z[pos_dst])
        neg_pred = model['link_pred'](z[neg_src], z[neg_dst])

        out_2d.fill_(torch.nan)
        out_2d[pos_src, pos_dst] = pos_pred.squeeze(-1).detach()
        out_2d[neg_src, neg_dst] = neg_pred.squeeze(-1).detach()
        out_2d.fill_diagonal_(0)  # Model does not predict self-loop edges

        # Valid number assertion
        assert torch.all(out_2d != torch.nan)

        # Visualize outputs
        if split_mode == "test":
            vis(out_2d, evalA, str(i))

        # MRR evaluation
        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            ).long()

            pred = model['link_pred'](z[src], z[dst])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([pred[0].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(pred[1:].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics, out_2d

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()

print("INFO: Arguments:", args)

DATA = args.data
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
if "num_nodes" in data_np:
    num_nodes = data_np["num_nodes"]
else:
    num_nodes = data_np["node_feat"].shape[0]
train_mask = list(data_np["train_mask"])
val_mask = list(data_np["val_mask"])
test_mask = list(data_np["test_mask"])

max_t = t.max()

############### NODE FEATURE GENERATION #################
node_feat = NodeFeatGenerator(args.node_feat, EMB_DIM)(num_nodes)
node_feat = node_feat.to(device)
adj = torch.zeros((max_t + 1, num_nodes, num_nodes))
adj[t, src, dst] = 1

val_start_t = t[val_mask].min()
test_start_t = t[test_mask].min()

# Separating train/val/test edge indices. "1/-1" is added to let each set also predicts the output of last day of week
train_adj = adj[:val_start_t]
val_adj = adj[val_start_t: test_start_t]
test_adj = adj[test_start_t:]

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
        args.normalize),
        inp_dim=node_feat.size(1))
link_pred = LinkPredictor(EMB_DIM, num_layers=2)

model = {'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
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
        loss, A = train()
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
        perf_metric_val, A = test(neg_sampler, split_mode="val", out_2d=A, start_time=val_start_t)
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
    perf_metric_test, _ = test(neg_sampler, split_mode="test", out_2d=A, start_time=test_start_t)

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
                  'num_units': NUM_UNITS,
                  'embedding_dim': EMB_DIM,
                  'node_feat': args.node_feat,
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
