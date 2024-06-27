"""
Trainer class
"""

from abc import ABC, abstractmethod
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import timeit
from typing import Any, Dict, List
import os
import os.path as osp
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

from ....tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from ....tgb.utils.utils import set_random_seed, save_results
from ....tgb.linkproppred.evaluate import Evaluator
from ....tse import MultiLayerEGCNO, EvolveGCNParams, LinkPredictor, NodeFeatType, NodeFeatGenerator, visualizer
from ....modules.early_stopping import  EarlyStopMonitor


class Trainer(ABC):
    def __init__(self, model_name: str):
        # Set arguments
        self.args: argparse.Namespace = self._get_args()
        print("INFO: Arguments:", self.args)

        # Data loading
        self.data_loading()

        # Set model
        self.model: Dict[str, torch.nn.Module] = self.get_model()
        self.model.__name__ = model_name

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set optimizer
        self.optim = self.get_optimizer()

        # Set objective
        self.criterion = self.get_criterion()

        # Set negative sampler
        self.neg_sampler = self.get_neg_sampler()

        # Set evaluator
        self.evaluator = self.get_evaluator()

        self.epoch: int = None

    @property
    def node_feat(self) -> torch.Tensor:
        return self._node_feat

    @property
    def train_adj(self) -> torch.Tensor:
        return self._train_adj
    
    @property
    def val_adj(self) -> torch.Tensor:
        return self._val_adj
    
    @property 
    def test_adj(self) -> torch.Tensor:
        return self._test_adj

    @property
    def src(self) -> torch.Tensor:
        return self._src
    
    @property
    def dst(self) -> torch.Tensor:
        return self._dst
    
    @property
    def t(self) -> torch.Tensor:
        return self._t
    
    @property
    def edge_feat(self) -> torch.Tensor:
        return self._edge_feat
    
    @property
    def num_nodes(self) -> torch.Tensor:
        return self._num_nodes
    
    @property
    def node_feat(self) -> torch.Tensor:
        return self._node_feat
    
    @property
    def val_start_t(self) -> int:
        return self._val_start_t
    
    @property
    def test_start_t(self) -> int:
        return self._test_start_t
    
    @property
    def results_path(self) -> str:
        return f'{osp.dirname(osp.abspath(__file__))}/saved_results'
    
    @property
    def first_metric(self) -> str:
        """ First metric to pick the best model based on """
        return "mrr"
    
    @property
    def model_params(self) -> List[str]:
        return self._model_params

    def _get_visualization_dir(self, split_mode: str) -> str:
        os.path.join(
            self.results_path, 
            self.model.__name__, 
            self.args.data, 
            split_mode, 
            f"NODEFEAT-{self.args.node_feat}_UNIT-{self.args.num_units}_EMB-{self.args.in_channels}", str(self.epoch)
        )
    
    def _get_results_filename(self) -> str:
        return f'{self.results_path}/{self.model.__name__}_{self.args.data}_results.json'
    
    @abstractmethod
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        pass
    
    @staticmethod
    def _set_running_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('-d', '--data', type=str, help='Dataset name')
        parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
        parser.add_argument('--k-value', type=int, help='k_value for computing ranking metrics', default=10)
        parser.add_argument('--num-epoch', type=int, help='Number of epochs', default=50)
        parser.add_argument('--node-feat', choices=NodeFeatType.list(), help='Type of node feature generation', default=NodeFeatType.CONSTANT)
        parser.add_argument('--seed', type=int, help='Random seed', default=1)
        parser.add_argument('--patience', type=float, help='Early stopper patience', default=5)
        parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
        parser.add_argument('--num-run', type=int, help='Number of iteration runs', default=1)
        parser.add_argument('-l', '--data-loc', type=str, help='The location where data is stored.')

        return parser
    
    def _get_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser('*** TGB ***')
        
        parser = self.set_model_args(parser)
        self._model_params = list()
        for action in parser._actions:
            self._model_params.append(action.dest)

        parser = self._set_running_args(parser)

        try:
            args = parser.parse_args()
        except:
            parser.print_help()
            sys.exit(0)
            
        return args

    def data_loading(self) -> None:
        """ This function generates the data for graph. """

        assert os.path.exists(self.args.data_loc), f"The given data location does not exist: {self.args.data_loc}"
        data_np = np.load(os.path.join(self.args.data_loc, self.args.data, "data.npz"))
        
        self._src = torch.tensor(data_np["src"])
        self._dst = torch.tensor(data_np["dst"])
        self._t = torch.tensor(data_np["t"])
        self._edge_feat = torch.tensor(data_np["edge_feat"])
        
        max_t = self.t.max()

        ############### NODE FEATURE GENERATION #################
        node_feat = NodeFeatGenerator(self.args.node_feat, self.args.in_channels)(self.num_nodes)
        self._node_feat = node_feat.to(self.device)

        if "num_nodes" in data_np:
            self._num_nodes = data_np["num_nodes"]
        else:
            self._num_nodes = data_np["node_feat"].shape[0]

        adj = torch.zeros((max_t + 1, self.num_nodes, self.num_nodes))
        adj[self.t, self.src, self.dst] = 1

        val_mask = list(data_np["val_mask"])
        test_mask = list(data_np["test_mask"])
        self._val_start_t = self.t[val_mask].min()
        self._test_start_t = self.t[test_mask].min()

        ############### SETTING ADJACENCY FOR DIFFERENT SETS ##############
        self._train_adj = adj[:self.val_start_t]
        self._val_adj = adj[self.val_start_t: self.test_start_t]
        self._test_adj = adj[self.test_start_t:]

    def get_neg_sampler(self):
        return NegativeEdgeSampler(dataset_name=self.args.data)

    def get_evaluator(self):
        return Evaluator(name=self.args.data)

    def _get_save_model_dir(self) -> str:
        return f'{osp.dirname(osp.abspath(__file__))}/saved_models/'

    def _get_save_model_id(self) -> str:
        return f'{self.model.__name__}_{self.args.data}_{self.args.seed}_{self.run_idx}_{self.args.in_channels}_{self.args.num_units}_{self.args.node_feat}'
    
    @abstractmethod
    def get_model(self) -> Dict[str, torch.nn.Module]:
        pass

    @abstractmethod
    def get_optimizer(self) -> torch.optim.optimizer.Optimizer:
        pass
    
    @abstractmethod
    def get_criterion(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def train_for_one_epoch(self) -> None:
        pass

    @abstractmethod
    def eval_for_one_epoch(self, split_mode: str) -> None:
        pass

    @abstractmethod
    def early_stopping_checker(self, early_stopper) -> bool:
        pass

    @abstractmethod
    def add_val_test_info(self, info: Dict[str, Any]) -> None:
        pass

    def _one_run(self):
        print('-------------------------------------------------------------------------------')
        print(f"INFO: >>>>> Run: {self.run_idx} <<<<<")
        start_run = timeit.default_timer()

        # set the seed for deterministic results...
        torch.manual_seed(self.run_idx + self.args.seed)
        set_random_seed(self.run_idx + self.args.seed)

        # define an early stopper
        save_model_dir = self._get_save_model_dir()
        save_model_id = self._get_save_model_id()
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                        tolerance=self.args.tolerance, patience=self.args.patience)

        # ==================================================== Train & Validation
        # loading the validation negative samples
        self.neg_sampler.load_eval_set(os.path.join(self.args.data_loc, self.args.data, "val_ns.pkl"), split_mode="val")

        self.val_perf_list: Dict[str, List[float]] = dict()
        self.test_perf: Dict[str, float] = dict()

        train_times_l, val_times_l = [], []
        free_mem_l, total_mem_l, used_mem_l = [], [], []
        start_train_val = timeit.default_timer()

        for self.epoch in range(1, self.args.num_epoch + 1):
            # training
            start_epoch_train = timeit.default_timer()
            self.train_for_one_epoch()
            end_epoch_train = timeit.default_timer()
            print(f"\tTraining elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}")
            # checking GPU memory usage
            free_mem, used_mem, total_mem = 0, 0, 0
            if torch.cuda.is_available():
                print("DEBUG: device: {}".format(torch.cuda.get_device_name(0)))
                free_mem, total_mem = torch.cuda.mem_get_info()
                used_mem = total_mem - free_mem
                print("------------Epoch {}: GPU memory usage-----------".format(self.epoch))
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
            with torch.no_grad():
                self.eval_for_one_epoch(split_mode="val")
            end_val = timeit.default_timer()
            print(f"\tValidation: Elapsed time (s): {end_val - start_val: .4f}")
            val_times_l.append(end_val - start_val)

            # check for early stopping
            if self.early_stopping_checker(early_stopper):
                break

        train_val_time = timeit.default_timer() - start_train_val
        print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

        # ==================================================== Test
        # first, load the best model
        early_stopper.load_checkpoint(self.model)

        # loading the test negative samples
        self.neg_sampler.load_eval_set(os.path.join(self.args.data_loc, self.args.data, "test_ns.pkl"), split_mode="test")

        # final testing
        start_test = timeit.default_timer()
        with torch.no_grad():
            self.eval_for_one_epoch(split_mode="test")
        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        test_time = timeit.default_timer() - start_test
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
        
        ### SAVE INFO ###
        info = {'data': self.args.data,
                'model': self.model.__name__,
                'run': self.run_idx,
                'seed': self.args.seed,
                'train_times': train_times_l,
                'free_mem': free_mem_l,
                'total_mem': total_mem_l,
                'used_mem': used_mem_l,
                'max_used_mem': max(used_mem_l),
                'val_times': val_times_l,
                'test_time': test_time,
                'train_val_total_time': np.sum(np.array(train_times_l)) + np.sum(np.array(val_times_l))}
        
        for p in self.model_params:
            info[p] = getattr(self.args, p)
        self.add_val_test_info(info)
        save_results(info
                    ,self._get_results_filename())

        print(f"INFO: >>>>> Run: {self.run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
        print('-------------------------------------------------------------------------------')

    def run(self):
        start_overall = timeit.default_timer()

        for self.run_idx in range(self.args.num_runs):
            self._one_run()
        print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
        print("==============================================================")
