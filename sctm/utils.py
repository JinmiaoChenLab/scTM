import numpy as np
import torch
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from scipy.sparse import issparse


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


def precompute_SGC(data, n_layers, add_self_loops=True):
    # Why is this inplace?

    edge_weight = data.edge_weight
    num_nodes = data.num_nodes
    assert data.edge_index is not None
    row, col = data.edge_index

    adj_t = SparseTensor(
        row=col, col=row, sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    print("Precomputing neighborhood")

    edge_index = gcn_norm(  # yapf: disable
        adj_t,
        edge_weight,
        num_nodes,
        False,
        add_self_loops,
    )

    assert data.x is not None
    xs = data.x
    xs = torch.log(xs + 1)
    sgc = [xs]
    for i in range(n_layers):
        xs = matmul(edge_index, xs, reduce="add")
        sgc.append(xs)

    # data["x"] = data.x
    data["sgc_x"] = torch.cat(sgc, dim=1)
    # torch.concat(sgc, dim = 1)
    return data
