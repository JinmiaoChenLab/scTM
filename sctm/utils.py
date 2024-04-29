import math

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, diags, identity, issparse


def densify(adata, layer):
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


def sparsify(adata, layer):
    if layer is None:
        if issparse(adata.X):
            data = adata.X
        else:
            data = csr_matrix(adata.X)
    else:
        if issparse(adata.layers[layer]):
            data = adata.layers[layer]
        else:
            data = csr_matrix(adata.layers[layer])
    return data


def get_init_bg(data):
    data = data / data.sum(axis=1, keepdims=True)
    # data = np.log(data)
    means = torch.mean(data, axis=0)
    print("Computing background frequencies")
    # 0.03 * (1 / ms)
    return np.log(means + 1e-15)


def precompute_SGC_scipy(x, adj, n_layers, mode="sign", add_diag=True, csr=True):
    deg = adj.sum(axis=1)
    # x = x / x.sum(axis = 1, keepdims =True)
    if add_diag:
        deg = deg + 2
        adj = adj + identity(n=x.shape[0])

    deg = diags(deg.A1).power(-0.5)
    deg.data[deg.data == np.inf] = 0

    # adj_sparse = make_sparse_tensor(adj)
    adj = deg @ adj @ deg
    adj = make_sparse_tensor(adj)

    assert x is not None
    # xs = x
    ls = x.sum(dim=1, keepdims=True)
    ms = torch.median(x.sum(axis=1))
    xs = x / ls * ms
    # xs = torch.log(x + 1)
    sgc = [xs]
    for _ in range(n_layers):
        # xs = matmul(edge_index, xs, reduce="add")
        xs = torch.sparse.mm(adj, sgc[-1])
        sgc.append(xs)
    if mode == "sgc":
        sgc = [sgc[-1]]
    sgc_x = torch.cat(sgc, dim=1)
    sgc_x = torch.log(sgc_x + 1)
    return sgc_x  # , adj_sparse


def make_sparse_tensor(adj):
    adj = adj.astype("float32").tocoo()
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    adj = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return adj


def softmax_torch(x, weights=1, dim=-1):  # Assuming x has atleast 2 dimensions
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = weights * torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs


def corr(adata, topic_prop, beta, topic, topk=20, layer=None, method="pearson"):
    cmn_index = np.intersect1d(adata.obs_names, topic_prop.index)
    df1 = topic_prop.loc[cmn_index, [topic]]
    genes = beta.nlargest(topk, topic).index
    df2 = densify(adata[cmn_index, genes], layer=layer)
    df2 = pd.DataFrame(df2, index=cmn_index, columns=genes)
    return df2.corrwith(df1[topic])


def nmf_init(adata, layer, n_topics):
    from sklearn.decomposition._nmf import _initialize_nmf

    counts = sparsify(adata, layer=layer)
    W, H = _initialize_nmf(counts, n_topics, init="nndsvd")
    # H = H / H.sum(axis=1, keepdims=True)
    # print(H.shape)
    H = np.log(H + 1e-16)
    H = H.astype("float32")
    return torch.from_numpy(H)


def rbf_kernel_batch(data, variance, lengthscale):
    # Compute pair-wise distances
    data = data[..., None]
    sq_dist = torch.cdist(data, data, p=1.0)  # Pairwise Euclidean distances
    abs_dist = torch.abs(sq_dist)
    # Compute RBF kernel values
    # print()
    # rbf_values = variance[..., None, None] * torch.exp(
    #     -sq_dist.pow(2) / (2 * lengthscale[..., None, None] ** 2)
    # )
    matern_values = (
        variance[..., None, None]
        * (1 + math.sqrt(3) * abs_dist / lengthscale[..., None, None])
        * torch.exp(-math.sqrt(3) * abs_dist / lengthscale[..., None, None])
    )

    return matern_values
