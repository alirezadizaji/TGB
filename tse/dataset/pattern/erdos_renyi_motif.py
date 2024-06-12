import networkx as nx
import numpy as np

from ...decorators import counter

def _erdos_renyi_motif(n: int, p: float, motif: str, motif_param: dict) -> nx.Graph:
    G: nx.Graph = nx.erdos_renyi_graph(n, p)

    func = getattr(nx, motif)
    G2:nx.Graph = func(**motif_param)
    motif_num_nodes = G2.number_of_nodes()

    # Pick nodes with smallest degrees to attach motif
    degrees = G.degree
    n1_id1 = np.sort(degrees)[:motif_num_nodes]
    
    # To attach G2 to G, reindex nodes within G2
    A2 = np.array(G2.edges)
    shape = A2.shape
    L = A2.flatten()
    L = n1_id1[L]
    A2 = A2.reshape(shape)

    G2 = G.copy()
    # Attach motif
    G.add_edges_from(A2)

    return G, G2

@counter(0)
def er_motif_weekly(**kwds):
    if er_motif_weekly.call == 0:
        G, G2 = _erdos_renyi_motif(**kwds)
        er_motif_weekly.G1 = G
        er_motif_weekly.G2 = G2
        
    if er_motif_weekly.call % 2 == 0:
        return er_motif_weekly.G1
    else:
        return er_motif_weekly.G2


