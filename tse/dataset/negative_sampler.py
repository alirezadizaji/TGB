from tqdm import tqdm
from typing import Optional

import numpy as np

def gen_neg_dst(src: np.ndarray, 
        dst: np.ndarray, 
        t: np.ndarray, 
        num_nodes: int, 
        num_neg_edges: Optional[int] = None) -> np.ndarray:
    """ It generates negative samples per each existing edge in a temporal graph """
    neg_dst = dict()
    n_id = np.arange(num_nodes)

    for pos_src, pos_dst, t_i in tqdm(zip(src, dst, t), total=len(src)):
        key = (pos_src, pos_dst, t_i)
        if key in neg_dst:
            raise Exception(f"Duplicated edge {key}")
        
        pos_dsts_same_src = dst[np.logical_and(src == pos_src, t_i == t)]
        nn_id = np.setdiff1d(n_id, pos_dsts_same_src)
        if num_neg_edges is None or num_neg_edges >= nn_id.size:
            neg_dst[key] = nn_id
        else:
            neg_dst[key] = np.random.choice(nn_id, size=num_neg_edges, replace=False)

    return neg_dst