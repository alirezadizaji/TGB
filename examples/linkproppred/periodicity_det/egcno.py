"""
Dynamic Link Prediction with a EvolveGCNO model with Early Stopping
"""

import argparse
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

from ....tse import MultiLayerEGCNO, EvolveGCNParams, LinkPredictor, visualizer
from .trainer import Trainer

class EvolveGCNTrainer(Trainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of EvolveGCN units', default=1)
        parser.add_argument('--in-channels', type=int, help='input channel dimension of EvolveGCNO', default=100)
        parser.add_argument('--improved', type=bool, help='If True, then identity is added to adjacency matrix', default=False)
        parser.add_argument('--cached', type=bool, help='If True, then EvolveGCN caches the normalized adjacency matrix and uses it in next steps', default=False)
        parser.add_argument('--normalize', type=bool, help='If True, then EvolveGCN normalizes the adjacency matrix', default=True)
        
        return parser
    
    def get_model(self) -> Dict[str, torch.nn.Module]:
        gnn = MultiLayerEGCNO(
            num_units=self.args.num_units, 
            base_args=EvolveGCNParams(
                self.args.in_channels,
                self.args.improved,
                self.args.cached,
                self.args.normalize),
                inp_dim=self.node_feat.size(1))
        link_pred = LinkPredictor(self.args.in_channels, num_layers=2)

        gnn.to(self.device)
        link_pred.to(self.device)

        return {'gnn': gnn,
                'link_pred': link_pred}

    def get_optimizer(self) -> torch.optim.optimizer.Optimizer:
        return torch.optim.Adam(
            set(self.model['gnn'].parameters()) | set(self.model['link_pred'].parameters()),
            lr=self.args.lr,
        )
    
    def get_criterion(self) -> torch.nn.Module:
        return torch.nn.BCELoss()

    def train_for_one_epoch(self) -> None:
        r"""
        Training procedure for EvolveGCN model
        This function uses some objects that are globally defined in the current scrips 

        Parameters:
            None
        Returns:
            None
                
        """
        # This variable stores the model output. All elements should have a valid number at the end
        out_2d = torch.zeros((self.num_nodes, self.num_nodes), requires_grad=False)

        self.model['gnn'].train()
        self.model['link_pred'].train()
        total_loss = 0
        for cur_t, trainA in enumerate(self.train_adj):
    
            prev_edge_index = None
            # At time step 0, the input graph is empty of edges
            if cur_t == 0:
                prev_src, prev_dst = torch.nonzero(torch.zeros_like(trainA), as_tuple=True)
            else:
                prev_src, prev_dst = torch.nonzero(self.train_adj[cur_t - 1], as_tuple=True)
            
            prev_edge_index = torch.stack([prev_src, prev_dst], dim=0)
            
            # Separate positive and negative pairs
            cur_src, cur_dst = torch.nonzero(trainA, as_tuple=True)

            # to prevent considering self-loop edges as negative pairs, temporarily set the diagonal value of trainA as nonzero
            trainA.fill_diagonal_(1)
            neg_src, neg_dst = torch.nonzero(trainA == 0, as_tuple=True)
            trainA.fill_diagonal_(0)

            prev_edge_index = prev_edge_index.to(self.device)
            self.optim.zero_grad()

            z = self.model['gnn'](
                    self.node_feat,
                    prev_edge_index.long())

            pos_pred: torch.Tensor = self.model['link_pred'](z[cur_src], z[cur_dst])
            neg_pred: torch.Tensor = self.model['link_pred'](z[neg_src], z[neg_dst])

            # Sigmoid assertion
            assert torch.all(pos_pred <= 1) and torch.all(pos_pred >= 0), "Sigmoid assertion failed. make sure `sigmoid` is applied at model output."

            # Loss calculation
            loss = self.criterion(pos_pred, torch.ones_like(pos_pred))
            loss += self.criterion(neg_pred, torch.zeros_like(neg_pred))

            loss.backward()
            self.optim.step()
            total_loss += float(loss.detach())

            # Predicted edge index
            out_2d.fill_(torch.nan)
            out_2d[cur_src, cur_dst] = pos_pred.squeeze(-1).detach().cpu()
            out_2d[neg_src, neg_dst] = neg_pred.squeeze(-1).detach().cpu()
            out_2d.fill_diagonal_(0)  # Model does not predict self-loop edges

            # Valid number assertion
            assert torch.all(out_2d != torch.nan)

        loss = total_loss / self.train_adj.size(0)
        print(f"Epoch: {self.epoch:02d}, Loss: {loss:.4f}")
        self._out_2d = out_2d

    def eval_for_one_epoch(self, split_mode: str) -> None:
        self.model['gnn'].eval()
        self.model['link_pred'].eval()

        vis = visualizer(save_dir=self._get_visualization_dir(split_mode))

        if split_mode == "val":
            start_time = self.val_start_t
            adj = self.val_adj
        elif split_mode == "test":
            start_time = self.test_start_t
            adj = self.test_adj
        else:
            raise Exception()
        
        perf_list = {k: [] for k in ["mrr", "acc", "pre", "rec", "f1"]}

        out_2d = self._out_2d.detach()
        
        for i, evalA in enumerate(adj):
            cur_t = i + start_time

            prev_edge_index = None
            # At first step of evaluation, pass the last time recurrent model output as the first adjacency input
            if i == 0:
                prev_adj = (out_2d >= 0.5).long()
            else:
                prev_adj = adj[i - 1]
            prev_src, prev_dst = torch.nonzero(prev_adj, as_tuple=True)
            prev_edge_index = torch.stack([prev_src, prev_dst], dim=0)
            
            # Separate positive and negative pairs
            pos_src, pos_dst = torch.nonzero(evalA, as_tuple=True)

            # To prevent considering self-loop edges as negative pairs, set diagonal elements as zero.
            evalA.fill_diagonal_(1)
            neg_src, neg_dst = torch.nonzero(evalA == 0, as_tuple=True)
            evalA.fill_diagonal_(0)

            pos_t = self.t[self.t == cur_t]
            neg_batch_list = self.neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

            prev_edge_index = prev_edge_index.to(self.device)
            z = self.model['gnn'](
                    self.node_feat,
                    prev_edge_index)

            # Predicted edge index
            pos_pred = self.model['link_pred'](z[pos_src], z[pos_dst])
            neg_pred = self.model['link_pred'](z[neg_src], z[neg_dst])

            out_2d.fill_(torch.nan)
            out_2d[pos_src, pos_dst] = pos_pred.squeeze(-1).detach().cpu()
            out_2d[neg_src, neg_dst] = neg_pred.squeeze(-1).detach().cpu()
            out_2d.fill_diagonal_(0)  # Model does not predict self-loop edges

            # Valid number assertion
            assert torch.all(out_2d != torch.nan)

            # Visualize outputs
            if split_mode == "test":
                vis(out_2d, evalA, str(i))

            # MRR evaluation
            for idx, neg_batch in enumerate(neg_batch_list):
                src = torch.full((1 + len(neg_batch),), pos_src[idx], device=self.device)
                dst = torch.tensor(
                    np.concatenate(
                        ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                        axis=0,
                    ),
                    device=self.device,
                ).long()

                pred = self.model['link_pred'](z[src], z[dst])

                # compute MRR
                input_dict = {
                    "y_pred_pos": np.array([pred[0].squeeze(dim=-1).cpu()]),
                    "y_pred_neg": np.array(pred[1:].squeeze(dim=-1).cpu()),
                    "eval_metric": [self.first_metric],
                }
                perf_list["mrr"].append(self.evaluator.eval(input_dict)[self.first_metric])

            # Binary classification evaluation
            out_1d = (out_2d.flatten() >= 0.5)
            eval_1d = evalA.flatten()
            perf_list["acc"].append(accuracy_score(eval_1d, out_1d))
            perf_list["pre"].append(precision_score(eval_1d, out_1d))
            perf_list["rec"].append(recall_score(eval_1d, out_1d))
            perf_list["f1"].append(f1_score(eval_1d, out_1d))

        self._perf_metrics = {k: np.mean(v) for k, v in perf_list.items()}
        self._out_2d = out_2d
        
        print(f"\tValidation {self.first_metric}: {self._perf_metrics[self.first_metric]: .4f}")

        if split_mode == "val":
            for k, v in self._perf_metrics.items():
                if k not in self.val_perf_list:
                    self.val_perf_list[k] = list()
                self.val_perf_list[k].append(v)
        else:
            for k, v in self._perf_metrics.items():
                self.test_perf[k] = float(v)

    def early_stopping_checker(self, early_stopper) -> bool:
        # check for early stopping
        if early_stopper.step_check(self._perf_metric[self.first_metric], self.model):
            return True
        return False  
    
    def add_val_test_info(self, info: Dict[str, Any]) -> None:
        info[f'val {self.first_metric}'] = self.val_perf_list[self.first_metric]

        for k, v in self.test_perf.items():
            info[f'test {k}'] = v