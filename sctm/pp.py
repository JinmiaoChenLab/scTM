import scanpy as sc
import numpy as np
from .utils import check_layer


def filter_genes(adata, min_cutoff=0.01, max_cutoff=1, expression_cutoff_99q=0):

    n_obs = adata.shape[0]
    min_cells = round(n_obs * min_cutoff)
    max_cells = round(n_obs * max_cutoff)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, max_cells=max_cells)

    data = check_layer(adata, layer=None)

    pass_cutoff = np.quantile(data, q=0.99, axis=0) > expression_cutoff_99q
    genes = adata.var_names[pass_cutoff]
    adata._inplace_subset_var(genes)
