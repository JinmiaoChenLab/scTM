import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import textwrap
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib import rcParams

# from upsetplot import plot, from_contents
from itertools import chain


from scanpy._utils import _empty, Empty
from scanpy.pl._tools.scatterplots import (
    _check_spatial_data,
    _check_img,
    _check_spot_size,
    _check_scale_factor,
    _check_crop_coord,
    _check_na_color,
)
from typing import (
    Union,
    Optional,
    List,
    Tuple,
)
from scanpy.pl._utils import (
    ColorLike,
)


def heatmap_topic(adata, groupby=None, topics=None, figsize=(10, 5), cmap=None):

    topic_prop = adata.obs.copy()

    if topics is None:
        topics = topic_prop.columns[topic_prop.columns.str.startswith("Topic")]
        topic_prop = topic_prop.loc[:, topics]
    else:
        topic_prop = topic_prop.loc[:, topics]

    topic_adata = ad.AnnData(topic_prop)
    sc.pl.clustermap(topic_adata, obs_key=groupby, figsize=figsize, cmap=cmap)


def heatmap(
    adata,
    groupby=None,
    topics=None,
    figsize=(10, 5),
    dendrogram=False,
    swap_axes=True,
    cmap=None,
    save=None,
    **kwargs,
):

    topic_prop = adata.obs.copy()

    if topics is None:
        topics = topic_prop.columns[topic_prop.columns.str.startswith("Topic")]
        topic_prop = topic_prop.loc[:, topics]
    else:
        topic_prop = topic_prop.loc[:, topics]

    topic_adata = ad.AnnData(topic_prop)
    topic_adata.obs[groupby] = adata.obs[groupby]
    sc.pl.heatmap(
        adata=topic_adata,
        var_names=topics,
        groupby=groupby,
        figsize=figsize,
        dendrogram=dendrogram,
        swap_axes=swap_axes,
        cmap=cmap,
        save=save,
        **kwargs,
    )


def matrixplot(
    adata,
    groupby=None,
    topics=None,
    figsize=(10, 5),
    dendrogram=False,
    swap_axes=True,
    cmap=None,
    save=None,
):

    topic_prop = adata.obs.copy()

    if topics is None:
        topics = topic_prop.columns[topic_prop.columns.str.startswith("Topic")]
        topic_prop = topic_prop.loc[:, topics]
    else:
        topic_prop = topic_prop.loc[:, topics]

    topic_adata = ad.AnnData(topic_prop)
    topic_adata.obs[groupby] = adata.obs[groupby]
    sc.pl.matrixplot(
        adata=topic_adata,
        var_names=topics,
        groupby=groupby,
        figsize=figsize,
        dendrogram=dendrogram,
        swap_axes=swap_axes,
        cmap=cmap,
        save=save,
    )


def tracksplot(
    adata,
    groupby=None,
    topics=None,
    figsize=(10, 5),
    dendrogram=False,
    swap_axes=True,
    cmap=None,
    save=None,
):

    topic_prop = adata.obs.copy()

    if topics is None:
        topics = topic_prop.columns[topic_prop.columns.str.startswith("Topic")]
        topic_prop = topic_prop.loc[:, topics]
    else:
        topic_prop = topic_prop.loc[:, topics]

    topic_adata = ad.AnnData(topic_prop)
    topic_adata.obs[groupby] = adata.obs[groupby]
    sc.pl.tracksplot(
        adata=topic_adata,
        var_names=topics,
        groupby=groupby,
        figsize=figsize,
        dendrogram=dendrogram,
        swap_axes=swap_axes,
        cmap=cmap,
        save=save,
    )


def clustermap(
    beta,
    gene_dict=None,
    topn_genes=20,
    xticklabels="auto",
    yticklabels="auto",
    figsize=(10, 6),
    cmap="viridis",
    fontsize=5,
    row_cluster=False,
    col_cluster=True,
    standard_scale=0,
    transpose=False,
    return_fig=False,
):

    if gene_dict is None:
        genes = []
        topics = beta.columns
        for i in topics:
            genes.append(beta.nlargest(topn_genes, i).index.tolist())
        genes = list(set(list(chain.from_iterable(genes))))
        beta_sub = beta.loc[genes, :]
    else:
        genes = [x for x in gene_dict.values()]
        genes = list(chain.from_iterable(genes))
        beta_sub = beta.loc[genes, :]

    if transpose:
        beta_sub = beta_sub.transpose()

    fig = sns.clustermap(
        beta_sub,
        cmap=cmap,
        figsize=figsize,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        standard_scale=standard_scale,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )

    fig.fig.subplots_adjust(right=0.7)
    fig.ax_cbar.set_position((0.8, 0.4, 0.01, 0.3))

    fig.ax_heatmap.set_yticklabels(
        fig.ax_heatmap.get_ymajorticklabels(), fontsize=fontsize, rotation=0
    )

    # fig.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
    # fig.ax_col_dendrogram.set_visible(False) #suppress column dendrogram

    if return_fig:
        fig = fig.fig
        return fig


def heatmap_topic_correlation(
    topic_prop,
    spatial_connectivities=None,
    return_values=False,
    figsize=(8, 6),
    cmap="viridis",
    fontsize=8,
):

    fig, ax = plt.subplots(figsize=figsize)

    if spatial_connectivities is None:
        corr = topic_prop.corr()
        sns.heatmap(
            corr,
            annot=True,
            vmin=-1,
            vmax=1,
            cmap=cmap,
            ax=ax,
            annot_kws={"fontsize": fontsize},
            fmt=".2f",
        )
    else:
        spatial_topic_prop = spatial_connectivities @ topic_prop
        spatial_topic_prop = pd.DataFrame(
            spatial_topic_prop, index=topic_prop.index, columns=topic_prop.columns
        )
        corr = spatial_topic_prop.corr()
        sns.heatmap(
            corr,
            annot=True,
            vmin=-1,
            vmax=1,
            cmap=cmap,
            ax=ax,
            annot_kws={"fontsize": fontsize},
            fmt=".2f",
        )
    if return_values:

        return corr
    else:
        return ax


def enrichment_barplot(
    enrichments,
    topic,
    type="enrichr",
    figsize=(10, 5),
    n_enrichments=5,
    qval_cutoff=0.05,
    title="auto",
):

    if type == "enrichr":

        if title == "auto":
            title = enrichments[topic]["Gene_set"][0]
        enrichment = enrichments[topic]
        enrichment = enrichment.loc[enrichment["Adjusted P-value"] < qval_cutoff, :]
        enrichment = enrichment.sort_values("Adjusted P-value")
        enrichment = enrichment.iloc[:n_enrichments, :]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            y=enrichment["Term"],
            width=-np.log(enrichment["Adjusted P-value"]),
            fill="blue",
            align="center",
        )

        ax.set_yticklabels(
            [textwrap.fill(term, 24) for term in enrichment["Term"].values]
        )

        ax.set_xlabel("- Log Adjusted P-value")
        ax.set_title(title)

        ax.invert_yaxis()

        plt.tight_layout()
        return ax

    if type == "gsea":

        if title == "auto":
            title = enrichments[topic]["Name"][0]

        enrichment = enrichments[topic]
        enrichment = enrichment.loc[enrichment["NOM p-val"] < qval_cutoff, :]
        enrichment = enrichment[enrichment["NES"] > 0]
        enrichment = enrichment.sort_values("NES", ascending=False)
        enrichment["Term"] = enrichment["Term"].str.replace("_", " ")
        enrichment = enrichment.iloc[:n_enrichments, :]

        enrichment["-log_qval"] = -np.log(
            enrichment["FDR q-val"].astype("float") + 1e-7
        )

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(y=enrichment["Term"], width=enrichment["NES"], align="center")

        ax.set_xlabel("NES")
        ax.set_title(title)

        ax.set_yticklabels(
            [textwrap.fill(term, 24) for term in enrichment["Term"].values]
        )

        ax.invert_yaxis()
        plt.tight_layout()

        return ax


def enrichment_dotplot(
    enrichment,
    type="enrichr",
    figsize=(10, 5),
    n_enrichments=10,
    title="auto",
    cmap=None,
):

    fig, ax = plt.subplots(figsize=figsize)

    if type == "enrichr":
        # enrichment = enrichments[topic].copy()
        # enrichment = enrichment.loc[enrichment["Adjusted P-value"] < qval_cutoff, :]

        enrichment["gene_size"] = enrichment["Overlap"].str.split("/").str[1]
        enrichment["-log_qval"] = -np.log(enrichment["Adjusted P-value"])
        enrichment["gene_ratio"] = enrichment["Overlap"].str.split("/").str[0].astype(
            "int"
        ) / enrichment["Overlap"].str.split("/").str[1].astype("int")

        if enrichment.shape[0] < n_enrichments:
            n_enrichments = enrichment.shape[0]

        enrichment = enrichment.sort_values("gene_ratio")
        enrichment = enrichment.iloc[:n_enrichments, :]

        scatter = ax.scatter(
            x=enrichment["gene_ratio"].values,
            y=enrichment["Term"].values,
            s=enrichment["gene_size"].values.astype("int"),
            c=enrichment["Combined Score"].values,
            cmap=cmap,
        )
        ax.set_xlabel("Gene Ratio")

        legend1 = ax.legend(
            *scatter.legend_elements(prop="sizes", num=5),
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            title="Geneset Size",
            labelspacing=1,
            borderpad=1,
        )
        ax.legend(
            *scatter.legend_elements(prop="colors", num=5),
            bbox_to_anchor=(1.04, 0),
            loc="lower left",
            title="Combined Score",
            labelspacing=1,
            borderpad=1,
        )

        ax.add_artist(legend1)
        # ax.add_artist(legend2)

        ax.set_yticklabels(
            [textwrap.fill(term, 24) for term in enrichment["Term"].values]
        )

        if title == "auto":
            ax.set_title(enrichment["Gene_set"].values[0])

    if type == "gsea":
        # enrichment = enrichments[topic].copy()
        # enrichment = enrichment.loc[enrichment["FDR q-val"] < qval_cutoff, :]

        enrichment["gene_size"] = enrichment["Tag %"].str.split("/").str[1]
        enrichment["-log_qval"] = -np.log(
            enrichment["FDR q-val"].astype("float") + 1e-7
        )
        enrichment["gene_ratio"] = enrichment["Tag %"].str.split("/").str[0].astype(
            "int"
        ) / enrichment["Tag %"].str.split("/").str[1].astype("int")

        if enrichment.shape[0] < n_enrichments:
            n_enrichments = enrichment.shape[0]

        enrichment = enrichment.sort_values("-log_qval", ascending=False)
        enrichment = enrichment.iloc[:n_enrichments, :]

        scatter = ax.scatter(
            x=enrichment["-log_qval"].values,
            y=enrichment["Term"].values,
            s=enrichment["gene_ratio"].values.astype("float"),
            c=enrichment["NES"].values,
            cmap=cmap,
        )
        ax.set_xlabel("-log q_val")

        legend1 = ax.legend(
            *scatter.legend_elements(prop="sizes", num=5),
            bbox_to_anchor=(1, 1),
            loc="upper left",
            title="Gene Ratio",
            labelspacing=1,
            borderpad=1,
        )

        ax.legend(
            *scatter.legend_elements(prop="colors", num=5),
            bbox_to_anchor=(1, 0),
            loc="lower left",
            title="NES",
            labelspacing=1,
            borderpad=1,
        )

        ax.add_artist(legend1)
        # ax.add_artist(legend2)

        ax.set_yticklabels(
            [textwrap.fill(term, 30) for term in enrichment["Term"].values]
        )

        if title == "auto":
            ax.set_title(enrichment["Name"].values[0])

        ax.invert_yaxis()
        plt.tight_layout()

        return ax


def draw_pie(dist, xpos, ypos, size, colors, figsize, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    c = 0
    for r1, r2 in zip(pie[:-1], pie[1:]):
        # if r2 - r1 > 0.01:
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, num=10)
        x = [0] + np.cos(angles).tolist() + [0]
        y = [0] + np.sin(angles).tolist() + [0]

        xy = np.column_stack([x, y])
        # print(xy.shape)
        ax.plot(
            [xpos],
            [ypos],
            marker=xy,
            markersize=np.abs(xy).max() * np.array(np.sqrt(size)),
            c=colors[c],
        )
        c += 1

    return ax


def spatialpie(
    adata,
    topic_prop,
    *,
    basis: str = "spatial",
    img: Union[np.ndarray, None] = None,
    img_key: Union[str, None, Empty] = _empty,
    library_id: Union[str, Empty] = _empty,
    crop_coord: Tuple[int, int, int, int] = None,
    alpha_img: float = 1.0,
    bw: Optional[bool] = False,
    frameon=False,
    size: float = 1.0,
    scale_factor: Optional[float] = None,
    spot_size: Optional[float] = None,
    na_color: Optional[ColorLike] = None,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    cmap="tab20",
    legend=True,
    figsize=None,
    title=None,
    **kwargs,
) -> Union[Axes, List[Axes], None]:
    """
    Scatter plot in spatial coordinates.
    This function allows overlaying data on top of images.
    Use the parameter `img_key` to see the image in the background
    And the parameter `library_id` to select the image.
    By default, `'hires'` and `'lowres'` are attempted.
    Use `crop_coord`, `alpha_img`, and `bw` to control how it is displayed.
    Use `size` to scale the size of the Visium spots plotted on top.
    As this function is designed to for imaging data, there are two key assumptions
    about how coordinates are handled:
    1. The origin (e.g `(0, 0)`) is at the top left – as is common convention
    with image data.
    2. Coordinates are in the pixel space of the source image, so an equal
    aspect ratio is assumed.
    If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`
    and `library_id` parameters to find values for `img`, `scale_factor`,
    and `spot_size` arguments. Alternatively, these values be passed directly.
    Parameters
    ----------
    {adata_color_etc}
    {scatter_spatial}
    {scatter_bulk}
    {show_save_ax}
    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    Examples
    --------
    This function behaves very similarly to other embedding plots like
    :func:`~scanpy.pl.umap`
    >>> adata = sc.datasets.visium_sge("Targeted_Visium_Human_Glioblastoma_Pan_Cancer")
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pl.spatial(adata, color="log1p_n_genes_by_counts")
    See Also
    --------
    :func:`scanpy.datasets.visium_sge`
        Example visium data.
    :tutorial:`spatial/basic-analysis`
        Tutorial on spatial analysis.
    """
    # get default image params if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    spot_size = _check_spot_size(spatial_data, spot_size)
    scale_factor = _check_scale_factor(
        spatial_data, img_key=img_key, scale_factor=scale_factor
    )
    crop_coord = _check_crop_coord(crop_coord, scale_factor)
    na_color = _check_na_color(na_color, img=img)

    if bw:
        cmap_img = "gray"
    else:
        cmap_img = None

    if scale_factor is not None:
        circle_radius = size * scale_factor * spot_size * 0.5
    else:
        circle_radius = spot_size * 0.5

    if figsize is None:
        figsize = (rcParams["figure.figsize"][0], rcParams["figure.figsize"][1])

    topic_names = topic_prop.columns
    topic_prop = topic_prop.to_numpy()

    n_cells = topic_prop.shape[0]
    axs = None
    spatial_coords = adata.obsm[basis]
    n = 0
    colors = plt.get_cmap(cmap)
    colors = colors.colors
    for i in range(n_cells):
        n = n + 1
        if (n % 20000) == 0:
            print(f"{i} number of cells done")
        axs = draw_pie(
            topic_prop[
                i,
            ],
            xpos=spatial_coords[i, 0] * scale_factor,
            ypos=spatial_coords[i, 1] * scale_factor,
            figsize=figsize,
            size=circle_radius,
            ax=axs,
            colors=colors,
        )

    if legend:
        legend_elements = []
        for i in range(len(topic_names)):
            legend_elements.append(
                Patch(facecolor=colors[i], edgecolor="w", label=topic_names[i])
            )

        axs.legend(
            handles=legend_elements,
            bbox_to_anchor=(1, 0.5),
            loc="center left",
            labelspacing=0.5,
            borderpad=0.5,
            frameon=frameon,
        )

    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
        if img is not None:
            ax.imshow(img, cmap=cmap_img, alpha=alpha_img)
        else:
            ax.set_aspect("equal")
            ax.invert_yaxis()
        if crop_coord is not None:
            ax.set_xlim(crop_coord[0], crop_coord[1])
            ax.set_ylim(crop_coord[3], crop_coord[2])
        else:
            ax.set_xlim(cur_coords[0], cur_coords[1])
            ax.set_ylim(cur_coords[3], cur_coords[2])

    ax.set_xlabel(basis + "1")
    ax.set_ylabel(basis + "2")

    ax.set_xticks([])
    ax.set_yticks([])

    if not frameon:
        ax.axis("off")

    if title is not None:
        ax.set_title(title)

    if show is False or return_fig is True:
        return axs
