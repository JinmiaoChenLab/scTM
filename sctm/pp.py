import numpy as np
import pandas as pd
import scanpy as sc

from .utils import densify, sparsify


def filter_genes(
    adata, min_cutoff=0.01, max_cutoff=1, expression_cutoff_99q=0, layer=None
):
    """Similar function to sc.pp.filter_genes but uses percentage instead of counts.
    Args:
        adata (_type_): Anndata
        min_cutoff (float, optional): Minimum percentage of counts required for a
            gene to pass filtering. Defaults to 0.01.
        max_cutoff (int, optional): Maximum percentage of counts required for a
            gene to pass filtering. Defaults to 0.01.. Defaults to 1.
        expression_cutoff_99q (int, optional): Minimum expression level of gene at
            99th percentile. Defaults to 0.
    """
    n_obs = adata.shape[0]
    min_cells = round(n_obs * min_cutoff)
    max_cells = round(n_obs * max_cutoff)

    data = sparsify(adata, layer=layer)
    counts = data.copy()
    counts.data = np.ones_like(counts.data)

    keep_genes = (counts.sum(axis=0) >= min_cells) & (counts.sum(axis=0) <= max_cells)
    genes = adata.var_names[keep_genes.A1]

    if expression_cutoff_99q > 0:
        pass_cutoff = (
            np.quantile(densify(adata, layer=layer), q=0.99, axis=0)
            > expression_cutoff_99q
        )
        genes = adata.var_names[keep_genes.A1 & pass_cutoff]

    adata._inplace_subset_var(genes)


def filter_cells(adata, min_genes=None, min_counts=None, layer=None):
    if min_genes is not None:
        data = sparsify(adata, layer=layer)
        counts = data.copy()
        counts.data = np.ones_like(counts.data)
        keep_cells = counts.sum(axis=1) >= min_genes
        cells = adata.obs_names[keep_cells.A1]
        adata._inplace_subset_obs(cells)

    if min_counts is not None:
        data = sparsify(adata, layer=layer)
        keep_cells = data.sum(axis=1) >= min_counts
        cells = adata.obs_names[keep_cells.A1]
        adata._inplace_subset_obs(cells)


def batch_highly_variable_genes(
    adata, batch_key, n_top_genes, layer=None, subset=False
):
    """Similar function to sc.pp.highly_variable_genes but fixes what I think its a bug
    in the implementation. Uses flavor seurat_v3 only.

    Args:
        adata (_type_): Anndata object
        batch_key (_type_): Batch key
        n_top_genes (_type_): _description_
        layer (_type_, optional): _description_. Defaults to None.
        subset (bool, optional): _description_. Defaults to False.
    """
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    nvars = adata.shape[1]
    adatas = [
        adata[adata.obs[batch_key] == cat]
        for cat in adata.obs[batch_key].cat.categories
    ]
    for a in adatas:
        sc.pp.highly_variable_genes(
            a,
            flavor="seurat_v3",
            n_top_genes=nvars,
            layer=layer,
            subset=False,
        )
    ranks = [adata.var.highly_variable_rank.values for adata in adatas]
    ranks = np.vstack(ranks)
    ranks = pd.DataFrame(ranks.transpose(), index=adata.var_names)
    adata.var["highly_variable_rank"] = ranks.median(axis=1)
    cutoff = adata.var["highly_variable_rank"].sort_values()[n_top_genes]
    adata.var["highly_variable"] = True
    adata.var.loc[adata.var.highly_variable_rank >= cutoff, "highly_variable"] = False
    genes = adata.var_names[adata.var.highly_variable]
    if subset:
        adata._inplace_subset_var(genes)
