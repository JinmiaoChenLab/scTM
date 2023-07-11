import numpy as np
import itertools
from scipy.sparse import issparse
import pandas as pd
from scipy.stats import spearmanr


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
    return data


def get_topic_diversity(beta, topk=10):
    num_topics = beta.shape[1]
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    genes = list(itertools.chain(*topics))
    n_unique = len(np.unique(genes))
    TD = n_unique / (topk * num_topics)
    return TD


def get_cell_probability(data, wi, quantiles_df, wj=None):

    if wj is None:
        D_wi = (data[wi] >= np.max((quantiles_df.loc[wi, "quantiles"], 1))).mean()
        return D_wi

    # Find probability that they are not both zero
    D_wj = (data[wj] >= np.max((quantiles_df.loc[wj, "quantiles"], 1))).mean()
    D_wi_wj = (
        (data[wi] >= np.max((quantiles_df.loc[wi, "quantiles"], 1)))
        & (data[wj] >= np.max((quantiles_df.loc[wj, "quantiles"], 1)))
    ).mean()

    return D_wj, D_wi_wj


def get_topic_coherence(
    adata, beta, layer=None, topk=10, quantile=0.8, individual=False
):

    data = check_layer(adata, layer)
    data = data.copy()
    data = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    quantiles = np.quantile(data, q=quantile, axis=0)
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
                    f_wi_wj = (np.log(D_wi_wj) - np.log(D_wi) - np.log(D_wj)) / (
                        -np.log(D_wi_wj + 1e-5)
                    )
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


def get_gene_topic_coherence(
    adata, beta, topic_prop, topk=10, layer=None, individual=False
):

    data = check_layer(adata, layer)

    data = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    # Drop batch
    topics = [beta.nlargest(topk, i).index.tolist() for i in beta.columns]
    TGC = []
    for i in range(len(topics)):
        topic_genes = topics[i]
        TGC_topic = []
        for topic_gene in topic_genes:
            cor = spearmanr(data[topic_gene], topic_prop.iloc[:, i])[0]
            TGC_topic.append(cor)
        TGC.append(np.mean(TGC_topic))

    if not individual:
        return np.mean(TGC)
    else:
        return TGC


def get_metrics(adata, beta, topic_prop, topk=10, layer=None, TGC=True):

    TC = get_topic_coherence(adata, beta, layer=layer, topk=topk)
    TD = get_topic_diversity(beta, topk=topk)
    if TGC:
        TGC = get_gene_topic_coherence(adata, beta, topic_prop, topk, layer)
        return {
            "Topic Coherence": TC,
            "Topic Diversity": TD,
            "Gene Topic Coherence": TGC,
        }
    else:
        return {"Topic Coherence": TC, "Topic Diversity": TD}
