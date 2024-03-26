import itertools

import numpy as np
import pandas as pd

# from scipy.stats import spearmanr
from numpy.linalg import norm

# import rbo
from .rbo import rbo_min
from .utils import densify


def get_topic_diversity(beta, topk=20):
    num_topics = beta.shape[1]
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    genes = list(itertools.chain(*topics))
    n_unique = len(np.unique(genes))
    TD = n_unique / (topk * num_topics)
    return TD


def get_cell_probability(data, wi, quantiles_df, wj=None, max=1):
    if wj is None:
        D_wi = (data[wi] >= np.max((quantiles_df.loc[wi, "quantiles"], max))).mean()
        return D_wi

    # Find probability that they are not both zero
    D_wj = (data[wj] >= np.max((quantiles_df.loc[wj, "quantiles"], max))).mean()
    D_wi_wj = (
        (data[wi] >= np.max((quantiles_df.loc[wi, "quantiles"], max)))
        & (data[wj] >= np.max((quantiles_df.loc[wj, "quantiles"], max)))
    ).mean()

    return D_wj, D_wi_wj


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


def get_topic_coherence(
    adata, beta, layer=None, topk=20, quantile=0.75, individual=False
):
    data = densify(adata, layer)
    data = data.copy()
    data = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    quantiles = np.quantile(data, q=quantile, axis=0)
    # quantiles = np.mean(data, axis=0)
    quantiles_df = pd.DataFrame(quantiles, index=adata.var_names, columns=["quantiles"])

    TC = []
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    for beta_topk in topics:
        TC_k = 0
        counter = 0
        for i, gene in enumerate(beta_topk):
            # get D(w_i)
            D_wi = get_cell_probability(data, gene, quantiles_df)
            j = i + 1
            tmp = 0
            while j < len(beta_topk) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_cell_probability(
                    data, gene, quantiles_df, beta_topk[j]
                )
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = 0
                else:
                    # f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (
                    #     np.log(D_wi_wj) - np.log(D)
                    # )
                    f_wi_wj = (np.log2(D_wi_wj) - np.log2(D_wi) - np.log2(D_wj)) / (
                        -np.log2(D_wi_wj)
                    )
                    # f_wi_wj = np.log((D_wi_wj + 1) / (D_wj))
                    # f_wi_wj = (np.log(D_wi) + np.log(D_wj)) / (np.log(D_wi_wj)) -1
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k / counter)

    if individual:
        return TC
    else:
        TC = np.mean(TC)
        return TC


def get_topic_gene_cosine(
    adata, beta, topic_prop, topk=20, layer=None, individual=False
):
    data = densify(adata, layer)
    data = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    # Drop batch
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    TGC = []
    for i in range(len(topics)):
        topic_genes = topics[i]
        TGC_topic = []
        for topic_gene in topic_genes:
            cos = cosine_similarity(data[topic_gene], topic_prop.iloc[:, i])
            TGC_topic.append(cos)
        TGC.append(np.mean(TGC_topic))

    if not individual:
        return np.mean(TGC)
    else:
        return TGC


def get_gene_topic_coherence(
    adata, beta, topic_prop, layer=None, topk=20, quantile=0.75, individual=False
):
    data = densify(adata, layer)
    data = data.copy()
    data = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    data = pd.concat([data, topic_prop], axis=1)

    quantiles = np.quantile(data, q=quantile, axis=0)
    quantiles_df = pd.DataFrame(
        quantiles,
        index=adata.var_names.tolist() + topic_prop.columns.tolist(),
        columns=["quantiles"],
    )

    TC = []
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    for beta_topk in topics:
        TC_k = 0
        counter = 0
        for i, gene in enumerate(beta_topk):
            # get D(w_i)
            D_wi = get_cell_probability(data, gene, quantiles_df, max=1e-8)
            # j = i + 1
            tmp = 0
            # while j < len(beta_topk) and j > i:
            for j in topic_prop.columns:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_cell_probability(
                    data, gene, quantiles_df, j, max=1e-8
                )
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = 0
                else:
                    # f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (
                    #     np.log(D_wi_wj) - np.log(D)
                    # )
                    f_wi_wj = (np.log2(D_wi_wj) - np.log2(D_wi) - np.log2(D_wj)) / (
                        -np.log2(D_wi_wj)
                    )
                    # f_wi_wj = np.log((D_wi_wj + 1) / (D_wj))
                    # f_wi_wj = (np.log(D_wi) + np.log(D_wj)) / (np.log(D_wi_wj)) -1
                # update tmp:
                tmp += f_wi_wj
                # j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k / counter)

    if individual:
        return TC
    else:
        TC = np.mean(TC)
        return TC


def get_rbo(beta, topk=20, p=0.8, individual=False):
    num_topics = beta.shape[1]
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    rbos = []
    for i in range(num_topics):
        rbos_topics = []
        for j in range(num_topics):
            if i != j:
                rbos_topics.append(1 - rbo_min(topics[i], topics[j], p=p))
                # rbos_topics.append(1 - rbo.RankingSimilarity(topics[i], topics[j]).rbo(p = p))
        rbos.append(np.min(rbos_topics))
    if individual:
        return rbos
    else:
        return np.mean(rbos)


def sparseness_hoyer(topic_prop):
    """
    The sparseness of array x is a real number in [0, 1], where sparser array
    has value closer to 1. Sparseness is 1 iff the vector contains a single
    nonzero component and is equal to 0 iff all components of the vector are
    the same

    modified from Hoyer 2004: [sqrt(n)-L1/L2]/[sqrt(n)-1]

    adapted from nimfa package: https://nimfa.biolab.si/
    """
    from math import sqrt  # faster than numpy sqrt

    x = topic_prop.values
    eps = np.finfo(x.dtype).eps if "int" not in str(x.dtype) else 1e-9

    x = x / x.sum(axis=1, keepdims=True)

    n = x.size

    # measure is meant for nmf: things get weird for negative values
    if np.min(x) < 0:
        x -= np.min(x)

    # patch for array of zeros
    if np.allclose(x, np.zeros(x.shape), atol=1e-6):
        return 0.0

    L1 = abs(x).sum()
    L2 = sqrt(np.multiply(x, x).sum())
    sparseness_num = sqrt(n) - (L1 + eps) / (L2 + eps)
    sparseness_den = sqrt(n) - 1

    return sparseness_num / sparseness_den


def get_metrics(adata, beta, topic_prop, topk=20, layer=None, TGC=True):
    TC = get_topic_coherence(adata, beta, layer=layer, topk=topk)
    TD = get_topic_diversity(beta, topk=topk)
    TRBO = get_rbo(beta, topk=topk)
    TS = sparseness_hoyer(topic_prop)

    if TGC:
        TGC = get_topic_gene_cosine(adata, beta, topic_prop, topk=topk, layer=layer)
        return {
            "Module Coherence": TC,
            "Module Diversity": TD,
            "Module Diversity(RBO)": TRBO,
            "Gene Topic Coherence": TGC,
            "Topic Sparsity": TS,
        }
    else:
        return {
            "Module Coherence": TC,
            "Module Diversity": TD,
            "Topic Diversity(RBO)": TRBO,
            "Topic Sparsity": TS,
        }
