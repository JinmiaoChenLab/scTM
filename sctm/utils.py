import numpy as np
import torch
from scipy.sparse import issparse
from torch_geometric.utils import scatter, to_torch_csc_tensor


def check_layer(adata, layer):
    if layer is None:
        if issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X
    else:
        if issparse(adata.layers[layer]):
            data = adata.layers[layer].toarray()
        else:
            data = adata.layers[layer]
    return data.astype("float32")


def get_init_bg(data):
    # Compute the log background frequency of all words
    # sums = np.sum(data, axis=0)+1
    data = data.copy()
    data = data / data.sum(axis=1, keepdims=True)
    means = np.mean(data, axis=0)
    print("Computing background frequencies")
    return np.log(means + 1e-8)


def precompute_SGC(data, n_layers, mode="sign"):

    if n_layers >= 1:
        row, col = data.edge_index
        N = data.num_nodes

        edge_weight = data.edge_weight
        if edge_weight is None:
            edge_weight = torch.ones(data.num_edges, device=row.device)

        deg = scatter(edge_weight, col, dim_size=N, reduce="sum")
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        adj = to_torch_csc_tensor(data.edge_index, edge_weight, size=(N, N))
        adj_t = adj.t()

    assert data.x is not None
    xs = data.x
    xs = torch.log(xs + 1)
    sgc = [xs]
    if mode == "sign":
        for i in range(n_layers):
            # xs = matmul(edge_index, xs, reduce="add")
            xs = adj_t @ sgc[-1]
            sgc.append(xs)
        data["sgc_x"] = torch.cat(sgc, dim=1)
    else:
        for i in range(n_layers):
            xs = adj_t @ sgc[-1]
            sgc.append(xs)
        data["sgc_x"] = sgc[-1]
    # data["x"] = data.x
    return data["sgc_x"]
